"""CLI command: bin2vec train — train the Bin2Vec neural network."""

from __future__ import annotations

import click


@click.command()
@click.pass_context
@click.option(
    "--vocab-path",
    default="data/vocab.json",
    show_default=True,
    help="Tokenizer vocabulary JSON (built by `bin2vec preprocess`).",
)
@click.option(
    "--output-dir",
    default="data/checkpoints",
    show_default=True,
    help="Directory for checkpoints.",
)
@click.option("--epochs", default=50, show_default=True, type=int)
@click.option("--batch-size", default=64, show_default=True, type=int)
@click.option("--lr", default=3e-4, show_default=True, type=float)
@click.option("--weight-decay", default=1e-4, show_default=True, type=float)
@click.option(
    "--temperature",
    default=0.07,
    show_default=True,
    type=float,
    help="NT-Xent temperature τ.",
)
@click.option(
    "--embed-dim",
    default=128,
    show_default=True,
    type=int,
    help="Final function embedding dimension.",
)
@click.option(
    "--bb-hidden-dim",
    default=128,
    show_default=True,
    type=int,
    help="BasicBlockEncoder Transformer hidden dim.",
)
@click.option("--bb-num-layers", default=2, show_default=True, type=int)
@click.option("--bb-num-heads", default=4, show_default=True, type=int)
@click.option(
    "--gnn-hidden-dim",
    default=128,
    show_default=True,
    type=int,
    help="CFGEncoder GATv2 hidden dim.",
)
@click.option("--gnn-num-layers", default=3, show_default=True, type=int)
@click.option("--gnn-num-heads", default=4, show_default=True, type=int)
@click.option(
    "--max-seq-len",
    default=128,
    show_default=True,
    type=int,
    help="Max tokens per basic block.",
)
@click.option("--max-vocab-size", default=32_768, show_default=True, type=int)
@click.option("--grad-clip", default=1.0, show_default=True, type=float)
@click.option(
    "--num-workers",
    default=4,
    show_default=True,
    type=int,
    help="DataLoader worker processes.",
)
@click.option(
    "--device",
    default="cuda",
    show_default=True,
    help="Torch device string (cuda / cpu).",
)
@click.option(
    "--save-every",
    default=5,
    show_default=True,
    type=int,
    help="Save a checkpoint every N epochs.",
)
@click.option(
    "--resume",
    default=None,
    type=str,
    help="Path to a checkpoint to resume training from.",
)
def train(
    ctx: click.Context,
    vocab_path: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    temperature: float,
    embed_dim: int,
    bb_hidden_dim: int,
    bb_num_layers: int,
    bb_num_heads: int,
    gnn_hidden_dim: int,
    gnn_num_layers: int,
    gnn_num_heads: int,
    max_seq_len: int,
    max_vocab_size: int,
    grad_clip: float,
    num_workers: int,
    device: str,
    save_every: int,
    resume: str | None,
) -> None:
    """Train the Bin2Vec contrastive model.

    Requires `bin2vec assemble` (to produce data/dataset/*.parquet) and
    `bin2vec preprocess` (to produce the vocabulary JSON) to have been run
    first.
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from pathlib import Path
    from bin2vec.train.train import train as _train

    data_dir = ctx.obj["data_dir"]
    dataset_dir = Path(data_dir) / "dataset"

    _train(
        dataset_dir=dataset_dir,
        vocab_path=vocab_path,
        output_dir=output_dir,
        embed_dim=embed_dim,
        bb_hidden_dim=bb_hidden_dim,
        bb_num_layers=bb_num_layers,
        bb_num_heads=bb_num_heads,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_num_layers=gnn_num_layers,
        gnn_num_heads=gnn_num_heads,
        max_seq_len=max_seq_len,
        max_vocab_size=max_vocab_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        temperature=temperature,
        grad_clip=grad_clip,
        num_workers=num_workers,
        device_str=device,
        save_every=save_every,
        resume=resume,
    )
