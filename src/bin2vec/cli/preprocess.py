"""CLI command: bin2vec preprocess (stub)."""

from __future__ import annotations

import click


@click.command()
@click.pass_context
def preprocess(ctx: click.Context) -> None:
    """Preprocess dataset (stub — not yet implemented)."""
    click.echo("Preprocessing is not yet implemented.")
    click.echo("This will support: disassembly, CFG extraction, tokenization.")
