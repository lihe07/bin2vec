"""CLI command: bin2vec extract."""

from __future__ import annotations

from pathlib import Path

import click

from bin2vec.config.matrix import filter_matrix, load_matrix
from bin2vec.config.packages import load_packages
from bin2vec.utils.logging import setup_logging


@click.command()
@click.option("--packages", "pkg_names", multiple=True, help="Package names to extract (default: all)")
@click.option("--isa", "isas", multiple=True, help="ISAs to extract (default: all)")
@click.option("--compiler", "compilers", multiple=True, help="Compiler families (gcc, clang)")
@click.option("--opt-level", "opt_levels", multiple=True, help="Optimization levels")
@click.option("--packages-config", default="config/packages.yaml", help="Path to packages YAML")
@click.option("--matrix-config", default="config/compiler_matrix.yaml", help="Path to matrix YAML")
@click.option("--workers", type=int, default=None, help="Number of worker processes (default: all CPUs)")
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def extract(
    ctx: click.Context,
    pkg_names: tuple[str, ...],
    isas: tuple[str, ...],
    compilers: tuple[str, ...],
    opt_levels: tuple[str, ...],
    packages_config: str,
    matrix_config: str,
    workers: int | None,
    verbose: bool,
) -> None:
    """Extract functions from compiled ELF binaries."""
    setup_logging(verbose)
    data_dir = ctx.obj["data_dir"]

    all_packages = load_packages(Path(packages_config))
    all_configs = load_matrix(Path(matrix_config))

    if pkg_names:
        packages = [p for p in all_packages if p.name in pkg_names]
    else:
        packages = all_packages

    configs = filter_matrix(
        all_configs,
        isas=list(isas) if isas else None,
        compilers=list(compilers) if compilers else None,
        opt_levels=list(opt_levels) if opt_levels else None,
    )

    click.echo(f"Extracting from {len(packages)} packages × {len(configs)} configs")

    from bin2vec.extract.extractor import extract_all

    total = extract_all(
        packages=packages, configs=configs, data_dir=data_dir, max_workers=workers
    )
    click.echo(f"\nTotal functions extracted: {total}")
