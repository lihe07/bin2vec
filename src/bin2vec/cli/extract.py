"""CLI command: bin2vec extract."""

from __future__ import annotations

from pathlib import Path

import click

from bin2vec.config.matrix import filter_matrix, load_matrix
from bin2vec.config.packages import PackageConfig
from bin2vec.utils.logging import setup_logging


def _discover_built_packages(data_dir: str) -> list[PackageConfig]:
    """Discover packages from the builds directory structure.

    Walks data/builds/<category>/<name>/<version>/ to find what was built.
    """
    builds_root = Path(data_dir).resolve() / "builds"
    packages = []
    if not builds_root.exists():
        return packages

    for category_dir in sorted(builds_root.iterdir()):
        if not category_dir.is_dir():
            continue
        for name_dir in sorted(category_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            for version_dir in sorted(name_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                packages.append(PackageConfig(
                    name=name_dir.name,
                    category=category_dir.name,
                    version=version_dir.name,
                    atom=f"{category_dir.name}/{name_dir.name}-{version_dir.name}",
                ))

    return packages


@click.command()
@click.option("--packages", "pkg_names", multiple=True, help="Package names to extract (default: all)")
@click.option("--categories", "category_names", multiple=True, help="Categories to extract (default: all)")
@click.option("--isa", "isas", multiple=True, help="ISAs to extract (default: all)")
@click.option("--compiler", "compilers", multiple=True, help="Compiler families (gcc, clang)")
@click.option("--opt-level", "opt_levels", multiple=True, help="Optimization levels")
@click.option("--matrix-config", default="config/compiler_matrix.yaml", help="Path to matrix YAML")
@click.option("--workers", type=int, default=None, help="Number of worker processes (default: all CPUs)")
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def extract(
    ctx: click.Context,
    pkg_names: tuple[str, ...],
    category_names: tuple[str, ...],
    isas: tuple[str, ...],
    compilers: tuple[str, ...],
    opt_levels: tuple[str, ...],
    matrix_config: str,
    workers: int | None,
    verbose: bool,
) -> None:
    """Extract functions from compiled ELF binaries."""
    setup_logging(verbose)
    data_dir = ctx.obj["data_dir"]

    all_packages = _discover_built_packages(data_dir)
    all_configs = load_matrix(Path(matrix_config))

    packages = all_packages
    if pkg_names:
        packages = [p for p in packages if p.name in pkg_names]
    if category_names:
        packages = [p for p in packages if p.category in category_names]

    configs = filter_matrix(
        all_configs,
        isas=list(isas) if isas else None,
        compilers=list(compilers) if compilers else None,
        opt_levels=list(opt_levels) if opt_levels else None,
    )

    click.echo(f"Extracting from {len(packages)} packages x {len(configs)} configs")

    from bin2vec.extract.extractor import extract_all

    total = extract_all(
        packages=packages, configs=configs, data_dir=data_dir, max_workers=workers
    )
    click.echo(f"\nTotal functions extracted: {total}")
