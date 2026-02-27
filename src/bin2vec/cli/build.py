"""CLI command: bin2vec build."""

from __future__ import annotations

from pathlib import Path

import click

from bin2vec.config.matrix import filter_matrix, load_matrix
from bin2vec.config.packages import load_categories
from bin2vec.utils.logging import setup_logging


@click.command()
@click.option("--categories", "category_names", multiple=True, help="Categories to build (default: all from config)")
@click.option("--packages", "pkg_names", multiple=True, help="Package names to build (default: all discovered)")
@click.option("--isa", "isas", multiple=True, help="ISAs to build for (default: all)")
@click.option("--compiler", "compilers", multiple=True, help="Compiler families (gcc, clang)")
@click.option("--opt-level", "opt_levels", multiple=True, help="Optimization levels (e.g. -O2)")
@click.option("--categories-config", default="config/categories.yaml", help="Path to categories YAML")
@click.option("--matrix-config", default="config/compiler_matrix.yaml", help="Path to matrix YAML")
@click.option("--docker-dir", default="docker", help="Directory containing Dockerfile")
@click.option("--workers", default=4, help="Max parallel builds")
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def build(
    ctx: click.Context,
    category_names: tuple[str, ...],
    pkg_names: tuple[str, ...],
    isas: tuple[str, ...],
    compilers: tuple[str, ...],
    opt_levels: tuple[str, ...],
    categories_config: str,
    matrix_config: str,
    docker_dir: str,
    workers: int,
    verbose: bool,
) -> None:
    """Build packages across the compilation matrix using Gentoo Portage."""
    setup_logging(verbose)
    data_dir = ctx.obj["data_dir"]

    categories = load_categories(Path(categories_config))
    if category_names:
        categories = [c for c in categories if c in category_names]

    all_configs = load_matrix(Path(matrix_config))
    configs = filter_matrix(
        all_configs,
        isas=list(isas) if isas else None,
        compilers=list(compilers) if compilers else None,
        opt_levels=list(opt_levels) if opt_levels else None,
    )

    click.echo(f"Building from {len(categories)} categories x {len(configs)} configs")

    from bin2vec.build.orchestrator import run_builds

    failures = run_builds(
        categories=categories,
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
