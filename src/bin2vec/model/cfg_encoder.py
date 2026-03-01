"""CFGEncoder: Graph Neural Network over basic-block node embeddings.

Architecture
------------
Given a CFG where each node (basic block) already has an embedding from
:class:`~bin2vec.model.basic_block_encoder.BasicBlockEncoder`, this module
runs several rounds of message passing to produce a graph-level embedding.

We use a **Graph Attention Network v2 (GATv2)** from torch-geometric because:
* Attention lets the model weight predecessor blocks differently.
* v2 fixes the static-attention problem of the original GAT.
* It is robust to variable graph sizes and is well-tested.

After GNN layers, a **global mean-pooling** aggregator collapses the node
embeddings into a single graph-level vector which is then projected to the
final embedding space.

Input (via :class:`torch_geometric.data.Data` or :class:`~.Batch`)
------
x          : FloatTensor [num_nodes_total, node_feat_dim]
    Pre-computed basic-block embeddings (from BasicBlockEncoder).
edge_index : LongTensor  [2, num_edges_total]
    COO edge list; directed (src → dst in CFG control flow).
batch      : LongTensor  [num_nodes_total]
    Graph membership for each node (0..B-1).  Required for batched inference.

Output
------
graph_emb  : FloatTensor [B, out_dim]
    One embedding per function (graph).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# torch-geometric imports — guarded so that the module can be *imported*
# (for type-checking etc.) even without torch-geometric installed.
try:
    from torch_geometric.nn import GATv2Conv, global_mean_pool
except ImportError as _pyg_err:  # pragma: no cover
    raise ImportError(
        "torch-geometric is required for CFGEncoder. "
        "Install with: uv add torch-geometric"
    ) from _pyg_err


class CFGEncoder(nn.Module):
    """Encode a CFG (as a PyG graph) into a fixed-size function embedding.

    Parameters
    ----------
    node_feat_dim:
        Dimensionality of incoming node features (= BasicBlockEncoder.hidden_dim).
    hidden_dim:
        Hidden dimension inside the GATv2 layers.
    out_dim:
        Final function embedding dimension.
    num_layers:
        Number of GATv2 message-passing rounds.
    num_heads:
        Attention heads per GATv2 layer.
    dropout:
        Dropout on attention coefficients and node features.
    residual:
        If True, add residual connections between GATv2 layers.
    """

    def __init__(
        self,
        node_feat_dim: int = 128,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.dropout = nn.Dropout(dropout)

        # Input projection: align node features to hidden_dim (if different)
        if node_feat_dim != hidden_dim:
            self.input_proj: nn.Module = nn.Linear(node_feat_dim, hidden_dim)
        else:
            self.input_proj = nn.Identity()

        # Stack of GATv2 layers
        # Each layer outputs (num_heads * hidden_dim), then averaged over heads
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = hidden_dim
        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=in_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=False,  # average over heads → output dim = hidden_dim
                add_self_loops=True,
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim

        # Output projection to final embedding space
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """Compute function embeddings from a (batched) CFG.

        Args:
            x:           Node feature matrix [N, node_feat_dim].
            edge_index:  COO edge index [2, E].
            batch:       Node → graph assignment [N].

        Returns:
            Graph-level embedding [B, out_dim].
        """
        # Project input to hidden_dim
        h = self.input_proj(x)  # [N, hidden_dim]

        # GATv2 message passing
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)  # [N, hidden_dim]
            h_new = norm(h_new)
            h_new = torch.relu(h_new)
            h_new = self.dropout(h_new)
            if self.residual:
                h = h + h_new  # residual
            else:
                h = h_new

        # Global pooling: mean over all nodes of each graph
        graph_emb = global_mean_pool(h, batch)  # [B, hidden_dim]

        # Final projection + normalisation
        graph_emb = self.out_proj(graph_emb)  # [B, out_dim]
        graph_emb = self.out_norm(graph_emb)

        return graph_emb
