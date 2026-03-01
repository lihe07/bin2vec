"""Bin2Vec training script.

Usage (via UV)
--------------
    uv run python -m bin2vec.train.train \\
        --dataset-dir data/dataset \\
        --vocab-path data/vocab.json \\
        --output-dir data/checkpoints \\
        --epochs 50 \\
        --batch-size 64 \\
        --lr 3e-4 \\
        --temperature 0.07 \\
        --embed-dim 128 \\
        --device cuda

Vocabulary bootstrap
--------------------
If ``--vocab-path`` does not exist, the trainer first builds the vocabulary
by scanning the training split (one pass over raw_bytes → CFG → tokens).
This can take several minutes for large datasets.  Save and reuse the JSON.

Checkpoints
-----------
Every epoch (or every ``--save-every`` steps) saves:
    <output-dir>/
        epoch_<N>.pt     — model state dict + config
        best.pt          — copy of the best val-loss checkpoint
        vocab.json       — tokenizer vocabulary (copy of --vocab-path)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from bin2vec.model.bin2vec import Bin2VecConfig, Bin2VecModel
from bin2vec.preprocess.cfg import extract_cfg, CFGGraph, BasicBlock
from bin2vec.preprocess.tokenizer import Tokenizer
from bin2vec.train.dataset import make_dataloader
from bin2vec.train.loss import NTXentLoss

logger = logging.getLogger("bin2vec.train")


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------


def build_vocab(
    train_parquet: str | Path,
    max_vocab_size: int = 32_768,
    sample_limit: int | None = None,
) -> Tokenizer:
    """Build a :class:`Tokenizer` vocabulary by scanning training data.

    Args:
        train_parquet:  Path to ``train.parquet``.
        max_vocab_size: Cap on vocabulary size.
        sample_limit:   If set, only use the first N rows (for quick testing).

    Returns:
        A :class:`Tokenizer` with the full instruction vocabulary.
    """
    import pyarrow.parquet as pq

    logger.info("Building vocabulary from %s ...", train_parquet)
    table = pq.read_table(str(train_parquet), columns=["raw_bytes", "isa"])
    if sample_limit is not None:
        table = table.slice(0, sample_limit)

    tokenizer = Tokenizer(max_vocab_size=max_vocab_size)
    total = table.num_rows
    for i, batch in enumerate(table.to_batches(max_chunksize=512)):
        for raw_bytes, isa in zip(
            batch.column("raw_bytes").to_pylist(),
            batch.column("isa").to_pylist(),
        ):
            if raw_bytes is None or len(raw_bytes) < 4:
                continue
            try:
                cfg = extract_cfg(bytes(raw_bytes), isa)
            except Exception:
                continue
            for bb in cfg.blocks:
                for token in bb.instructions:
                    tokenizer.add_token(token)
        if i % 10 == 0:
            logger.info(
                "  scanned ~%d / %d rows, vocab size %d",
                min((i + 1) * 512, total),
                total,
                tokenizer.vocab_size,
            )

    logger.info("Vocabulary built: %d tokens", tokenizer.vocab_size)
    return tokenizer


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------


def run_epoch(
    model: Bin2VecModel,
    loader,
    criterion: NTXentLoss,
    optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    train: bool = True,
) -> float:
    """Run one epoch and return average loss."""
    model.train(train)
    total_loss = 0.0
    total_batches = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_a, batch_b, _keys in tqdm(
            loader, leave=False, desc="train" if train else "val"
        ):
            batch_a = batch_a.to(device)
            batch_b = batch_b.to(device)

            # Forward pass for both views
            emb_a = model(batch_a.x, batch_a.edge_index, batch_a.batch)
            emb_b = model(batch_b.x, batch_b.edge_index, batch_b.batch)

            # L2 normalise before loss
            emb_a = F.normalize(emb_a, dim=-1)
            emb_b = F.normalize(emb_b, dim=-1)

            loss = criterion(emb_a, emb_b)

            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: Bin2VecModel,
    path: Path,
    epoch: int,
    val_loss: float,
    config_dict: dict,
) -> None:
    """Save model weights + metadata to *path*."""
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "bin2vec_config": config_dict,
        },
        path,
    )


def load_checkpoint(model: Bin2VecModel, path: Path, device: torch.device) -> dict:
    """Load weights from *path* into *model*.  Returns the checkpoint dict."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(
    dataset_dir: str | Path,
    vocab_path: str | Path,
    output_dir: str | Path,
    # Model hyperparameters
    embed_dim: int = 128,
    bb_hidden_dim: int = 128,
    bb_num_layers: int = 2,
    bb_num_heads: int = 4,
    gnn_hidden_dim: int = 128,
    gnn_num_layers: int = 3,
    gnn_num_heads: int = 4,
    max_seq_len: int = 128,
    max_vocab_size: int = 32_768,
    # Training hyperparameters
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    temperature: float = 0.07,
    grad_clip: float = 1.0,
    num_workers: int = 4,
    # Misc
    device_str: str = "cuda",
    save_every: int = 5,
    resume: str | Path | None = None,
) -> None:
    dataset_dir = Path(dataset_dir)
    vocab_path = Path(vocab_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------
    if vocab_path.exists():
        logger.info("Loading vocabulary from %s", vocab_path)
        tokenizer = Tokenizer.load(vocab_path)
    else:
        tokenizer = build_vocab(
            dataset_dir / "train.parquet",
            max_vocab_size=max_vocab_size,
        )
        tokenizer.save(vocab_path)
        logger.info("Vocabulary saved to %s", vocab_path)

    # Copy vocab to output dir for reproducibility
    shutil.copy2(vocab_path, output_dir / "vocab.json")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    config = Bin2VecConfig(
        vocab_size=tokenizer.vocab_size,
        bb_hidden_dim=bb_hidden_dim,
        bb_num_heads=bb_num_heads,
        bb_num_layers=bb_num_layers,
        bb_ffn_dim=bb_hidden_dim * 4,
        bb_max_seq_len=max_seq_len,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_out_dim=gnn_hidden_dim,
        gnn_num_layers=gnn_num_layers,
        gnn_num_heads=gnn_num_heads,
        embed_dim=embed_dim,
    )
    model = Bin2VecModel(config).to(device)
    logger.info("Model: %d trainable parameters", model.num_parameters())

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    criterion = NTXentLoss(temperature=temperature).to(device)

    # ------------------------------------------------------------------
    # Optionally resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    if resume is not None:
        ckpt = load_checkpoint(model, Path(resume), device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        logger.info("Resumed from epoch %d (val_loss=%.4f)", start_epoch, best_val_loss)

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    train_loader = make_dataloader(
        parquet_path=dataset_dir / "train.parquet",
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = make_dataloader(
        parquet_path=dataset_dir / "val.parquet",
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_workers=num_workers,
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info("Starting training for %d epochs", epochs)
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip=grad_clip,
            train=True,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            grad_clip=grad_clip,
            train=False,
        )
        scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | %.1fs",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
            elapsed,
        )

        # Save periodic checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = output_dir / f"epoch_{epoch + 1:04d}.pt"
            save_checkpoint(model, ckpt_path, epoch, val_loss, vars(config))
            logger.info("Checkpoint saved → %s", ckpt_path)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, output_dir / "best.pt", epoch, val_loss, vars(config)
            )
            logger.info("  ↑ New best val_loss=%.4f", best_val_loss)

    logger.info("Training complete.  Best val_loss=%.4f", best_val_loss)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Bin2Vec model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-dir",
        default="data/dataset",
        help="Directory with train/val/test.parquet",
    )
    p.add_argument(
        "--vocab-path",
        default="data/vocab.json",
        help="Path to vocabulary JSON (built if missing)",
    )
    p.add_argument(
        "--output-dir",
        default="data/checkpoints",
        help="Directory for checkpoints and logs",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--bb-hidden-dim", type=int, default=128)
    p.add_argument("--bb-num-layers", type=int, default=2)
    p.add_argument("--bb-num-heads", type=int, default=4)
    p.add_argument("--gnn-hidden-dim", type=int, default=128)
    p.add_argument("--gnn-num-layers", type=int, default=3)
    p.add_argument("--gnn-num-heads", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--max-vocab-size", type=int, default=32_768)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument(
        "--resume", default=None, help="Path to a checkpoint to resume training from"
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    train(
        dataset_dir=args.dataset_dir,
        vocab_path=args.vocab_path,
        output_dir=args.output_dir,
        embed_dim=args.embed_dim,
        bb_hidden_dim=args.bb_hidden_dim,
        bb_num_layers=args.bb_num_layers,
        bb_num_heads=args.bb_num_heads,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_num_layers=args.gnn_num_layers,
        gnn_num_heads=args.gnn_num_heads,
        max_seq_len=args.max_seq_len,
        max_vocab_size=args.max_vocab_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        device_str=args.device,
        save_every=args.save_every,
        resume=args.resume,
    )
