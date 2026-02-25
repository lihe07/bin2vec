"""Parallel build orchestrator for (package × config) builds."""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from bin2vec.build.build_scripts import generate_build_script
from bin2vec.build.docker_runner import DockerRunner
from bin2vec.config.matrix import CompilationConfig
from bin2vec.config.packages import PackageConfig
from bin2vec.utils.logging import get_logger
from bin2vec.utils.paths import builds_dir, sources_dir

log = get_logger("orchestrator")


def _detect_container_runtime() -> str:
    """Return 'podman' if available, else 'docker'."""
    for rt in ("podman", "docker"):
        if shutil.which(rt):
            return rt
    raise RuntimeError("No container runtime found (podman or docker)")


def _download_apt_source(pkg: PackageConfig, src_dir: Path) -> Path:
    """Download source via apt-get source inside a container."""
    extracted = src_dir / pkg.source_dir_name
    if extracted.exists():
        log.debug("Source already exists: %s", extracted)
        return extracted

    runtime = _detect_container_runtime()
    apt_pkg = pkg.source.package or pkg.name
    version = pkg.version

    # Script to run inside the container:
    # 1. Enable deb-src repos
    # 2. apt-get source the package
    # 3. Find and extract only the .orig.tar.* file
    script = f"""
set -e
cd /work

# Enable deb-src
sed -i '/^Types: deb$/s/$/ deb-src/' /etc/apt/sources.list.d/*.sources 2>/dev/null || true
apt-get update -qq

# Download source files
apt-get source --download-only {apt_pkg}={version}* 2>/dev/null || apt-get source --download-only {apt_pkg}

# Find the orig tarball (prefer .orig.tar.*, fall back to .tar.*)
orig=$(ls *.orig.tar.* 2>/dev/null | head -1)
if [ -z "$orig" ]; then
    orig=$(ls *.tar.* 2>/dev/null | head -1)
fi

if [ -z "$orig" ]; then
    echo "ERROR: No tarball found for {apt_pkg}" >&2
    exit 1
fi

# Extract it
mkdir -p /work/extracted
tar xf "$orig" -C /work/extracted --strip-components=0

echo "ORIG_TARBALL=$orig"
"""

    log.info("Fetching apt source for %s (version %s)", apt_pkg, version)

    # Use a temp dir for container output, then move the extracted source
    staging = src_dir / f".apt-staging-{pkg.name}"
    staging.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                runtime, "run", "--rm",
                "-v", f"{staging}:/work",
                "ubuntu:24.04",
                "bash", "-c", script,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            log.error("apt-get source failed for %s:\n%s", apt_pkg, result.stderr)
            raise RuntimeError(f"apt-get source failed for {apt_pkg}")

        log.debug("apt-get source output: %s", result.stdout)

        # Find extracted directory inside staging/extracted/
        extracted_staging = staging / "extracted"
        if not extracted_staging.exists():
            raise RuntimeError(f"No extracted dir found for {apt_pkg}")

        # There should be one top-level directory
        subdirs = [d for d in extracted_staging.iterdir() if d.is_dir()]
        if subdirs:
            source = subdirs[0]
        else:
            source = extracted_staging

        # Move to final location
        shutil.move(str(source), str(extracted))
        log.info("Extracted apt source to %s", extracted)

    finally:
        # Clean up staging
        shutil.rmtree(staging, ignore_errors=True)

    return extracted


def download_source(pkg: PackageConfig, data_dir: str) -> Path:
    """Download and extract source tarball (idempotent)."""
    src_dir = sources_dir(data_dir)
    src_dir.mkdir(parents=True, exist_ok=True)

    extracted = src_dir / pkg.source_dir_name
    if extracted.exists():
        log.debug("Source already exists: %s", extracted)
        return extracted

    if pkg.source.type == "apt":
        return _download_apt_source(pkg, src_dir)

    url = pkg.source.url
    if not url:
        raise ValueError(f"Package {pkg.name} has source type '{pkg.source.type}' but no URL")
    filename = url.split("/")[-1]
    tarball_path = src_dir / filename

    if not tarball_path.exists():
        log.info("Downloading %s", url)
        urllib.request.urlretrieve(url, tarball_path)

    log.info("Extracting %s", tarball_path)
    with tarfile.open(tarball_path) as tf:
        tf.extractall(path=src_dir)

    if not extracted.exists():
        # Some tarballs have different top-level dir names; find it
        members = []
        with tarfile.open(tarball_path) as tf:
            members = tf.getnames()
        if members:
            top = members[0].split("/")[0]
            actual = src_dir / top
            if actual.exists() and actual != extracted:
                actual.rename(extracted)

    return extracted


def _run_single_build(
    pkg: PackageConfig,
    config: CompilationConfig,
    data_dir: str,
    docker_runner: DockerRunner,
    docker_dir: Path,
) -> tuple[str, bool]:
    """Run a single build. Returns (config_tag, success)."""
    tag = config.config_tag
    out_dir = builds_dir(data_dir, pkg, config)

    # Resume support: skip if output already has files
    if out_dir.exists() and any(out_dir.iterdir()):
        log.debug("Skipping existing build: %s/%s", pkg.name, tag)
        return tag, True

    src_path = sources_dir(data_dir)
    script = generate_build_script(
        pkg, config,
        source_dir=pkg.source_dir_name,
        output_dir=str(out_dir),
    )

    docker_runner.ensure_image(config.isa.docker_image, docker_dir, config.isa.name)

    success, logs = docker_runner.run_build(
        image_name=config.isa.docker_image,
        build_script=script,
        sources_path=src_path,
        output_path=out_dir,
    )

    if not success:
        log.error("FAILED: %s / %s", pkg.name, tag)
        # Write build log for debugging
        log_file = out_dir / "build.log"
        log_file.write_text(logs)

    return tag, success


def run_builds(
    packages: list[PackageConfig],
    configs: list[CompilationConfig],
    data_dir: str,
    docker_dir: Path,
    max_workers: int = 4,
) -> dict[str, list[str]]:
    """Run all builds in parallel. Returns {pkg_name: [failed_tags]}."""
    # Download all sources first
    for pkg in packages:
        download_source(pkg, data_dir)

    runner = DockerRunner()
    failures: dict[str, list[str]] = {pkg.name: [] for pkg in packages}

    tasks = [(pkg, config) for pkg in packages for config in configs]
    total = len(tasks)

    log.info("Starting %d builds (%d packages × %d configs)", total, len(packages), len(configs))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_run_single_build, pkg, config, data_dir, runner, docker_dir): (pkg, config)
            for pkg, config in tasks
        }

        with tqdm(total=total, desc="Building", unit="build") as pbar:
            for future in as_completed(futures):
                pkg, config = futures[future]
                try:
                    tag, success = future.result()
                    if not success:
                        failures[pkg.name].append(tag)
                except Exception as e:
                    log.error("Build exception for %s/%s: %s", pkg.name, config.config_tag, e)
                    failures[pkg.name].append(config.config_tag)
                pbar.update(1)

    runner.close()

    failed_count = sum(len(v) for v in failures.values())
    log.info("Builds complete: %d/%d succeeded", total - failed_count, total)

    return failures
