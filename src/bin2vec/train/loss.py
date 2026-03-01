"""Contrastive losses for Bin2Vec training.

NT-Xent (Normalized Temperature-scaled Cross Entropy)
------------------------------------------------------
Also known as the SimCLR loss (Chen et al., 2020).

Given a batch of B function identities, each with two compiled-variant views
(z_a[i], z_b[i]), the loss for the pair (z_a[i], z_b[i]) is:

    L_i = -log( exp(sim(z_a[i], z_b[i]) / τ)
                / Σ_{j≠i} [ exp(sim(z_a[i], z_b[j]) / τ)
                           + exp(sim(z_a[i], z_a[j]) / τ) ] )

where sim(u,v) = u·v / (||u|| ||v||)  (cosine similarity).

This is computed symmetrically (both (a→b) and (b→a) directions) and averaged.

Triplet Loss
------------
For ablation studies or when you prefer a margin-based loss.

References
----------
Chen et al., "A Simple Framework for Contrastive Learning of Visual
Representations", ICML 2020.  https://arxiv.org/abs/2002.05709
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class NTXentLoss(nn.Module):
    """NT-Xent contrastive loss for paired function embeddings.

    Parameters
    ----------
    temperature:
        Scaling factor τ.  Lower = sharper distribution, harder negatives.
        Typical values: 0.05–0.2.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        """Compute NT-Xent loss.

        Args:
            z_a: FloatTensor [B, D] — embeddings for view A (L2-normalised).
            z_b: FloatTensor [B, D] — embeddings for view B (L2-normalised).

        Returns:
            Scalar loss.

        Note:
            Input embeddings should already be L2-normalised
            (``F.normalize(z, dim=-1)``).  If they are not, this method
            normalises them internally.
        """
        B = z_a.size(0)
        if B < 2:
            # Can't form negatives with a single sample
            return torch.tensor(0.0, device=z_a.device, requires_grad=True)

        # Ensure L2 normalisation
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        # Concatenate both views: [2B, D]
        z = torch.cat([z_a, z_b], dim=0)  # [2B, D]

        # Similarity matrix: [2B, 2B]
        sim = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pairs:
        #   For view A[i] at row i,   the positive is view B[i] at row B+i
        #   For view B[i] at row B+i, the positive is view A[i] at row i
        pos_idx = torch.arange(B, device=z.device)
        labels = torch.cat([pos_idx + B, pos_idx])  # [2B]

        loss = F.cross_entropy(sim, labels)
        return loss


class TripletLoss(nn.Module):
    """Online semi-hard triplet loss.

    Given a batch of function embeddings with identity labels, mines semi-hard
    triplets (anchor, positive, negative) and computes the triplet margin loss.

    Parameters
    ----------
    margin:
        Margin α in max(d(a,p) - d(a,n) + α, 0).
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """Compute semi-hard triplet loss.

        Args:
            embeddings: FloatTensor [N, D] — L2-normalised embeddings.
            labels:     LongTensor  [N]    — identity label indices.

        Returns:
            Scalar loss (mean over valid triplets).
        """
        embeddings = F.normalize(embeddings, dim=-1)

        # Pairwise squared Euclidean distance
        # ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u·v = 2 - 2 cos_sim  (for unit vecs)
        dot = torch.matmul(embeddings, embeddings.T)
        dist2 = 2.0 - 2.0 * dot  # [N, N]
        dist2 = dist2.clamp(min=0.0)

        # Masks
        labels = labels.unsqueeze(1)  # [N, 1]
        pos_mask = labels.eq(labels.T)  # [N, N] True where same identity
        neg_mask = ~pos_mask

        # Remove diagonal from pos_mask
        eye = torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
        pos_mask = pos_mask & ~eye

        # For each anchor: d(a,p) — hardest positive (largest d)
        # If no positive exists for this anchor, d_pos = 0
        d_pos = (dist2 * pos_mask.float()).max(dim=1).values  # [N]

        # For each anchor: d(a,n) — semi-hard negative
        #   semi-hard: d(a,n) > d(a,p) AND d(a,n) < d(a,p) + margin
        # Fall back to hardest negative if no semi-hard exists
        d_neg_all = dist2.clone()
        d_neg_all[~neg_mask] = float("inf")
        d_neg_hard = d_neg_all.min(dim=1).values  # [N]

        # Semi-hard: among negatives that are farther than d_pos
        semi_hard_mask = (
            neg_mask
            & dist2.gt(d_pos.unsqueeze(1))
            & dist2.lt((d_pos + self.margin).unsqueeze(1))
        )
        d_neg_semi = dist2.clone()
        d_neg_semi[~semi_hard_mask] = float("inf")
        has_semi_hard = semi_hard_mask.any(dim=1)  # [N]

        d_neg = torch.where(
            has_semi_hard,
            d_neg_semi.min(dim=1).values,
            d_neg_hard,
        )  # [N]

        # Triplet loss
        loss_per_anchor = F.relu(d_pos - d_neg + self.margin)

        # Only average over anchors that have at least one positive
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        loss = loss_per_anchor[has_pos].mean()
        return loss
