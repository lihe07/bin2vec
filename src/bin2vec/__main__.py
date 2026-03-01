"""CLI entry point for bin2vec."""

import click

from bin2vec.cli.build import build
from bin2vec.cli.extract import extract
from bin2vec.cli.assemble import assemble
from bin2vec.cli.preprocess import preprocess
from bin2vec.cli.train import train


@click.group()
@click.option("--data-dir", default="data", help="Root data directory")
@click.pass_context
def cli(ctx: click.Context, data_dir: str) -> None:
    """Bin2Vec: Build, extract, assemble, and train binary function fingerprints."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir


cli.add_command(build)
cli.add_command(extract)
cli.add_command(assemble)
cli.add_command(preprocess)
cli.add_command(train)

if __name__ == "__main__":
    cli()
