"""Parallel build orchestrator using Gentoo Portage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from bin2vec.build.docker_runner import DockerRunner
from bin2vec.build.gentoo import build_package, discover_packages, install_deps
from bin2vec.config.matrix import CompilationConfig
from bin2vec.config.packages import PackageConfig
from bin2vec.utils.logging import get_logger
from bin2vec.utils.paths import builds_dir

log = get_logger("orchestrator")


def _build_single(
    runner: DockerRunner,
    container_id: str,
    pkg: PackageConfig,
    config: CompilationConfig,
    data_dir: str,
) -> tuple[str, bool]:
    """Build a single (package, config) combination. Returns (config_tag, success)."""
    tag = config.config_tag
    out_dir = builds_dir(data_dir, pkg, config)

    # Resume: skip if output already has files
    if out_dir.exists() and any(out_dir.iterdir()):
        log.debug("Skipping existing build: %s/%s", pkg.cp, tag)
        return tag, True

    # Build via Portage with ROOT pointing to the output inside the container
    output_root = f"/workspace/output/{pkg.category}/{pkg.name}/{pkg.version}"
    success = build_package(runner, container_id, pkg, config, output_root)

    if success:
        # Copy built files from container to host
        container_out = f"{output_root}/{tag}"
        runner.copy_from_container(container_id, container_out, out_dir)

    if not success:
        out_dir.mkdir(parents=True, exist_ok=True)
        log.error("FAILED: %s / %s", pkg.cp, tag)

    return tag, success


def run_builds(
    categories: list[str],
    configs: list[CompilationConfig],
    data_dir: str,
    docker_dir: Path,
    max_workers: int = 4,
    binpkg_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Run all builds using a long-lived Gentoo container.

    Args:
        categories:   Portage categories to discover packages from.
        configs:      Compilation configurations (compiler × opt-level × ISA).
        data_dir:     Root data directory on the host.
        docker_dir:   Directory containing the Dockerfile.
        max_workers:  Number of parallel build threads.
        binpkg_dir:   Host directory to use as the Portage binary package cache
                      (``PKGDIR``).  Mounted into the container at
                      ``/var/cache/binpkgs``.  When supplied, already-compiled
                      packages are reused as ``.gpkg`` files and new packages
                      are written there, dramatically speeding up repeat builds.
                      Defaults to ``<data_dir>/binpkgs``.

    Returns:
        ``{pkg_name: [failed_tags]}``.
    """
    runner = DockerRunner()
    runner.ensure_image(docker_dir)

    output_path = Path(data_dir).resolve() / "builds"
    output_path.mkdir(parents=True, exist_ok=True)

    # Default binpkg cache location sits inside the data directory so it is
    # naturally co-located with the rest of the pipeline outputs.
    if binpkg_dir is None:
        binpkg_dir = Path(data_dir).resolve() / "binpkgs"

    container_id = runner.start_container(output_path, binpkg_dir=binpkg_dir)

    try:
        # Discover packages from categories inside the container
        packages = discover_packages(categories, runner, container_id)
        if not packages:
            log.error("No packages discovered, aborting")
            return {}

        log.info("Discovered %d packages", len(packages))

        # Collect unique ISAs from configs
        seen_isas = {}
        for config in configs:
            seen_isas[config.isa.name] = config.isa

        # Install build deps once per package per ISA.
        # With the binpkg cache mounted, deps that were already emerged on a
        # previous run are restored in seconds from their .gpkg files.
        for pkg in packages:
            for isa in seen_isas.values():
                install_deps(runner, container_id, pkg, isa)

        failures: dict[str, list[str]] = {pkg.name: [] for pkg in packages}
        tasks = [(pkg, config) for pkg in packages for config in configs]
        total = len(tasks)

        log.info(
            "Starting %d builds (%d packages x %d configs)",
            total,
            len(packages),
            len(configs),
        )

        # Parallel builds across packages (ThreadPoolExecutor)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _build_single, runner, container_id, pkg, config, data_dir
                ): (pkg, config)
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
                        log.error(
                            "Build exception for %s/%s: %s",
                            pkg.cp,
                            config.config_tag,
                            e,
                        )
                        failures[pkg.name].append(config.config_tag)
                    pbar.update(1)

        failed_count = sum(len(v) for v in failures.values())
        log.info("Builds complete: %d/%d succeeded", total - failed_count, total)

        return failures

    finally:
        runner.stop_container(container_id)
        runner.close()
