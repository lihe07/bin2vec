"""CLI command: bin2vec preprocess — build vocabulary from assembled dataset."""

from __future__ import annotations

from pathlib import Path

import click

from bin2vec.utils.logging import get_logger

logger = get_logger("bin2vec.cli.preprocess")


@click.command()
@click.pass_context
@click.option(
    "--vocab-path",
    default="data/vocab.json",
    show_default=True,
    help="Output path for the tokenizer vocabulary JSON.",
)
@click.option(
    "--max-vocab-size",
    default=32_768,
    show_default=True,
    type=int,
    help="Maximum number of instruction tokens in the vocabulary.",
)
@click.option(
    "--sample-limit",
    default=None,
    type=int,
    help="Only scan the first N rows of training data (for quick tests).",
)
def preprocess(
    ctx: click.Context,
    vocab_path: str,
    max_vocab_size: int,
    sample_limit: int | None,
) -> None:
    """Build instruction vocabulary from the assembled training dataset.

    Reads data/dataset/train.parquet, extracts CFGs from each function's
    raw bytes, and collects all normalised instruction tokens into a
    vocabulary file used by the Bin2Vec model.

    Run this after `bin2vec assemble` and before `bin2vec train`.
    """
    data_dir = ctx.obj["data_dir"]
    train_parquet = Path(data_dir) / "dataset" / "train.parquet"

    if not train_parquet.exists():
        click.echo(
            f"ERROR: {train_parquet} not found. Run `bin2vec assemble` first.",
            err=True,
        )
        raise SystemExit(1)

    vocab_path_obj = Path(vocab_path)
    if vocab_path_obj.exists():
        click.echo(f"Vocabulary already exists at {vocab_path}. Delete it to rebuild.")
        return

    # Import here to keep startup fast when this command is not used
    from bin2vec.train.train import build_vocab

    tokenizer = build_vocab(
        train_parquet=train_parquet,
        max_vocab_size=max_vocab_size,
        sample_limit=sample_limit,
    )
    tokenizer.save(vocab_path_obj)
    click.echo(f"Vocabulary ({tokenizer.vocab_size} tokens) saved → {vocab_path}")
