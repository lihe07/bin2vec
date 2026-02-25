"""CLI command: bin2vec assemble."""

from __future__ import annotations

import click

from bin2vec.utils.logging import setup_logging


@click.command()
@click.option("--min-variants", default=10, help="Minimum variants per function identity")
@click.option("--train-ratio", default=0.8, help="Train split ratio")
@click.option("--val-ratio", default=0.1, help="Validation split ratio")
@click.option("--seed", default=42, help="Random seed for splitting")
@click.option("-v", "--verbose", is_flag=True)
@click.pass_context
def assemble(
    ctx: click.Context,
    min_variants: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    verbose: bool,
) -> None:
    """Assemble extracted functions into train/val/test dataset."""
    setup_logging(verbose)
    data_dir = ctx.obj["data_dir"]

    from bin2vec.assemble.writer import assemble_dataset

    counts = assemble_dataset(
        data_dir=data_dir,
        min_variants=min_variants,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    click.echo(f"\nDataset assembled:")
    for split, count in counts.items():
        click.echo(f"  {split}: {count} rows")
