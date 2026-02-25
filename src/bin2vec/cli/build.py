"""CLI command: bin2vec build."""

from __future__ import annotations

from pathlib import Path

import click

from bin2vec.config.matrix import filter_matrix, load_matrix
from bin2vec.config.packages import load_packages
from bin2vec.utils.logging import setup_logging


@click.command()
@click.option("--packages", "pkg_names", multiple=True, help="Package names to build (default: all)")
@click.option("--isa", "isas", multiple=True, help="ISAs to build for (default: all)")
@click.option("--compiler", "compilers", multiple=True, help="Compiler families (gcc, clang)")
@click.option("--opt-level", "opt_levels", multiple=True, help="Optimization levels (e.g. -O2)")
@click.option("--packages-config", default="config/packages.yaml", help="Path to packages YAML")
@click.option("--matrix-config", default="config/compiler_matrix.yaml", help="Path to matrix YAML")
@click.option("--docker-dir", default="docker", help="Directory containing Dockerfiles")
@click.option("--workers", default=4, help="Max parallel builds")
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def build(
    ctx: click.Context,
    pkg_names: tuple[str, ...],
    isas: tuple[str, ...],
    compilers: tuple[str, ...],
    opt_levels: tuple[str, ...],
    packages_config: str,
    matrix_config: str,
    docker_dir: str,
    workers: int,
    verbose: bool,
) -> None:
    """Build packages across the compilation matrix."""
    setup_logging(verbose)
    data_dir = ctx.obj["data_dir"]

    all_packages = load_packages(Path(packages_config))
    all_configs = load_matrix(Path(matrix_config))

    # Filter packages
    if pkg_names:
        packages = [p for p in all_packages if p.name in pkg_names]
    else:
        packages = all_packages

    # Filter configs
    configs = filter_matrix(
        all_configs,
        isas=list(isas) if isas else None,
        compilers=list(compilers) if compilers else None,
        opt_levels=list(opt_levels) if opt_levels else None,
    )

    click.echo(f"Building {len(packages)} packages × {len(configs)} configs = {len(packages) * len(configs)} builds")

    from bin2vec.build.orchestrator import run_builds

    failures = run_builds(
        packages=packages,
        configs=configs,
        data_dir=data_dir,
        docker_dir=Path(docker_dir),
        max_workers=workers,
    )

    total_failed = sum(len(v) for v in failures.values())
    if total_failed:
        click.echo(f"\n{total_failed} builds failed:")
        for pkg_name, tags in failures.items():
            for tag in tags:
                click.echo(f"  {pkg_name}/{tag}")
    else:
        click.echo("\nAll builds succeeded.")
