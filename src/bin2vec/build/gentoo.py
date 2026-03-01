"""Gentoo Portage build logic: package discovery, env overrides, emerge commands."""

from __future__ import annotations

import re

from bin2vec.build.docker_runner import DockerRunner
from bin2vec.config.matrix import CompilationConfig, ISAConfig
from bin2vec.config.packages import PackageConfig
from bin2vec.utils.logging import get_logger

log = get_logger("gentoo")

# ---------------------------------------------------------------------------
# Portage binary-package flags
# ---------------------------------------------------------------------------
# buildpkg  – write a .gpkg binary package for every package emerged so it can
#             be reused on future runs without recompiling.
# getbinpkg – prefer installing from a pre-built .gpkg when one exists in
#             PKGDIR (mounted from the host via docker_runner).
#
# These flags are appended to FEATURES for every emerge invocation so that
# the cache is populated/consumed consistently across install_deps and
# build_package calls.
_BINPKG_FEATURES = "buildpkg getbinpkg binpkg-multi-instance"


def discover_packages(
    categories: list[str],
    runner: DockerRunner,
    container_id: str,
) -> list[PackageConfig]:
    """Discover available packages in given categories from the portage tree.

    Queries /var/db/repos/gentoo/<category>/ inside the container to find
    all available packages with their latest versions.
    """
    packages = []

    for category in categories:
        # List package directories in the category
        exit_code, output = runner.exec_in_container(
            container_id,
            ["ls", f"/var/db/repos/gentoo/{category}/"],
        )
        if exit_code != 0:
            log.warning("Failed to list category %s: %s", category, output)
            continue

        pkg_names = [
            name.strip()
            for name in output.strip().split("\n")
            if name.strip() and not name.strip().startswith(".")
        ]

        for pkg_name in pkg_names:
            # Skip metadata directory
            if pkg_name == "metadata.xml" or pkg_name == "Manifest":
                continue

            # Find the latest ebuild version
            exit_code, output = runner.exec_in_container(
                container_id,
                [
                    "bash",
                    "-c",
                    f"ls /var/db/repos/gentoo/{category}/{pkg_name}/*.ebuild 2>/dev/null | sort -V | tail -1",
                ],
            )
            if exit_code != 0 or not output.strip():
                log.debug("No ebuilds found for %s/%s", category, pkg_name)
                continue

            ebuild_path = output.strip()
            # Extract version from ebuild filename: package-version.ebuild
            ebuild_name = ebuild_path.split("/")[-1].replace(".ebuild", "")
            # The version is everything after the package name and a dash followed by a digit
            match = re.match(rf"^{re.escape(pkg_name)}-(\d.*)$", ebuild_name)
            if not match:
                log.debug("Could not parse version from %s", ebuild_name)
                continue

            version = match.group(1)
            atom = f"{category}/{pkg_name}-{version}"

            packages.append(
                PackageConfig(
                    name=pkg_name,
                    category=category,
                    version=version,
                    atom=atom,
                )
            )

    log.info(
        "Discovered %d packages across %d categories", len(packages), len(categories)
    )
    return packages


def write_env_file(
    runner: DockerRunner,
    container_id: str,
    config: CompilationConfig,
) -> str:
    """Write a /etc/portage/env/ file for the given compilation config.

    Returns the env file name (used in package.env mapping).
    """
    env_name = config.env_tag
    env_content = (
        f'CC="{config.cc}"\n'
        f'CXX="{config.cxx}"\n'
        f'CFLAGS="{config.cflags}"\n'
        f'CXXFLAGS="{config.cflags}"\n'
    )

    runner.exec_in_container(
        container_id,
        ["mkdir", "-p", "/etc/portage/env"],
    )
    runner.exec_in_container(
        container_id,
        [
            "bash",
            "-c",
            f"cat > /etc/portage/env/{env_name} << 'ENVEOF'\n{env_content}ENVEOF",
        ],
    )

    return env_name


def write_package_env(
    runner: DockerRunner,
    container_id: str,
    pkg: PackageConfig,
    env_name: str,
) -> None:
    """Write /etc/portage/package.env entry mapping a package to an env file."""
    entry = f"{pkg.cp} {env_name}\n"
    runner.exec_in_container(
        container_id,
        ["bash", "-c", f"echo '{entry}' > /etc/portage/package.env"],
    )


def install_deps(
    runner: DockerRunner,
    container_id: str,
    pkg: PackageConfig,
    isa: ISAConfig,
) -> bool:
    """Install build dependencies for a package (once per package per ISA).

    Dependencies are installed with ``buildpkg`` + ``getbinpkg`` so that:
    * Any dep compiled for the first time is cached as a .gpkg in PKGDIR.
    * On subsequent runs (or for packages that share deps) the .gpkg is
      installed directly without recompilation.
    """
    emerge_cmd = isa.emerge_cmd
    log.info("Installing deps for %s (%s)", pkg.cp, isa.name)

    exit_code, output = runner.exec_in_container(
        container_id,
        [
            "bash",
            "-c",
            f'FEATURES="{_BINPKG_FEATURES}" '
            f"{emerge_cmd} --onlydeps --autounmask-write=y --autounmask-continue=y "
            f"--usepkg -q {pkg.cp}",
        ],
        timeout=1800,
    )
    if exit_code != 0:
        log.warning(
            "Dep install failed for %s (%s): %s", pkg.cp, isa.name, output[-500:]
        )
        return False
    return True


def build_package(
    runner: DockerRunner,
    container_id: str,
    pkg: PackageConfig,
    config: CompilationConfig,
    output_root: str,
) -> bool:
    """Build a single package with the given compilation config.

    Uses emerge --oneshot --nodeps with ROOT set to the output directory.

    Binary package behaviour
    ------------------------
    * ``buildpkg`` writes a .gpkg under PKGDIR keyed by
      ``<category>/<name>-<ver>-<config_tag>.gpkg`` (Portage uses the env
      tag via the CONFIG_ROOT / PORTAGE_CONFIGROOT difference to distinguish
      slots built with different compiler flags).
    * ``getbinpkg`` allows reuse of an existing .gpkg if the package atom,
      USE flags, and CFLAGS hash all match.  Because each config_tag sets
      different CFLAGS/CC, each compiler × opt-level combination produces its
      own distinct binary package entry, so there is no cross-contamination.
    """
    env_name = write_env_file(runner, container_id, config)
    write_package_env(runner, container_id, pkg, env_name)

    emerge_cmd = config.isa.emerge_cmd
    root = f"{output_root}/{config.config_tag}"

    log.info("Building %s with %s", pkg.cp, config.config_tag)

    exit_code, output = runner.exec_in_container(
        container_id,
        [
            "bash",
            "-c",
            f'ROOT="{root}" '
            f'FEATURES="-sandbox -usersandbox -pid-sandbox -network-sandbox nostrip '
            f'{_BINPKG_FEATURES}" '
            f"{emerge_cmd} --oneshot --nodeps --usepkg -q {pkg.cp}",
        ],
        timeout=1800,
    )
    if exit_code != 0:
        log.error(
            "Build failed for %s/%s: %s", pkg.cp, config.config_tag, output[-500:]
        )
        return False

    log.info("Build succeeded: %s/%s", pkg.cp, config.config_tag)
    return True
