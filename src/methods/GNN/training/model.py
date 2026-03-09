import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GraphNorm, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import JumpingKnowledge

N_ENTITY_TYPES = 5

class PreAggregation(nn.Module):
    """
    7 independent linear projections (Section 5.2.1):
      - 1 for primary nodes 
      - 6 for entity nodes

    All nodes are projected to the same hidden_dim so that
    the subsequent GCN layers can treat them uniformly.
    """

    def __init__(self, psych_dim: int, hidden_dim: int, entity_dim: int):
        super().__init__()

        # One projection for primary nodes
        self.primary_proj = nn.Linear(psych_dim, hidden_dim)

        # Six projections — one per entity category
        # (same dim in, so we could use a ModuleList of identical shapes)
        self.entity_projs = nn.ModuleList([
            nn.Linear(entity_dim, hidden_dim)
            for _ in range(N_ENTITY_TYPES)
        ])

        self.psych_dim = psych_dim
        self.entity_dim = entity_dim

    def forward(self, x_primary: torch.Tensor, x_entities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_primary  : [B*2, psych_dim]          — primary node features for a batch
                          (2 primary nodes × B graphs)
            x_entities : [B*NUM_ENTITIES, entity_dim]        — entity node features
                          (NUM_ENTITIES entity nodes × B graphs), ordered as
                          [cand_e0, cand_e1, ..., cand_e5, job_e0, ..., job_e5]
                          repeated for each graph in the batch
        """
        x_primary_out = self.primary_proj(x_primary)          # [B*2, H]


        BN = x_entities.shape[0]
        B   = BN // (N_ENTITY_TYPES * 2)  # number of graphs in batch

        # Reshape to [B, 12, 2304]
        x_ent = x_entities.view(B, N_ENTITY_TYPES*2, self.entity_dim)

        proj_out = []
        for local_idx in range(N_ENTITY_TYPES*2):
            category = local_idx % N_ENTITY_TYPES   # 0..4
            proj_out.append(self.entity_projs[category](x_ent[:, local_idx, :]))  # [B, H]

        x_entities_out = torch.stack(proj_out, dim=1).view(BN, -1)  # [B*N, H]

        return x_primary_out, x_entities_out

class CJMGCN(nn.Module):
    """
    Full GCN pipeline for Candidate-Job Matching.

    Args:
        hidden_channels  (int)  : width of all hidden layers
        num_gc_layers    (int)  : number of GCN message-passing layers (1–5)
        dropout          (float): dropout rate (0.0 or 0.5)
        use_jumping_knowledge (bool): whether to use JK-Net aggregation
        num_readout_layers    (int): depth of the MLP head (0–3)
                                     0 = direct linear projection to output
    """

    def __init__(
        self,
        hidden_channels: int = 128,
        num_gc_layers: int = 2,
        dropout: float = 0.0,
        use_jumping_knowledge: bool = False,
        num_readout_layers: int = 1,
        psych_dim: int = 18,
        entity_dim: int = 2304,
):
        super().__init__()


        self.hidden_channels       = hidden_channels
        self.num_gc_layers         = num_gc_layers
        self.dropout               = dropout
        self.use_jumping_knowledge = use_jumping_knowledge
        self.num_readout_layers    = num_readout_layers

        # ── Pre-aggregation ──────────────────────────────────────────────
        self.pre_agg = PreAggregation(psych_dim, hidden_channels, entity_dim)

        # ── GCN layers ───────────────────────────────────────────────────
        self.convs   = nn.ModuleList()
        self.norms   = nn.ModuleList()

        for _ in range(num_gc_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(GraphNorm(hidden_channels))

        if use_jumping_knowledge:
            self.jk = JumpingKnowledge(mode='cat',
                                        channels=hidden_channels,
                                        num_layers=num_gc_layers)
            post_conv_dim = hidden_channels * num_gc_layers
        else:
            post_conv_dim = hidden_channels

        # ── Global Pooling → concatenate sum + mean + max ────────────────
        # Output: 3 × post_conv_dim
        pool_out_dim = post_conv_dim * 3

        # ── Deep Readout MLP ─────────────────────────────────────────────
        readout_layers = []
        in_dim = pool_out_dim
        for _ in range(num_readout_layers):
            readout_layers += [
                nn.Linear(in_dim, hidden_channels),
                nn.LeakyReLU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = hidden_channels
        self.readout_mlp = nn.Sequential(*readout_layers)

        # ── Output Head ──────────────────────────────────────────────────
        self.output_head = nn.Linear(in_dim, 1)

    # ────────────────────────────────────────────────────────────────────
    def forward(self, x_primary, x_entities, edge_index, edge_weight, batch):

        # 1. Project heterogeneous features → common dim
        x_prim, x_ent = self.pre_agg(x_primary, x_entities)

        # 2. Reconstruct full node feature matrix [B*(2 + N_ENTITY_TYPES*2), H]
        #    We need to interleave primary and entity nodes back into
        #    the original node ordering: [cand, job, ent_0..11] per graph.
        B = x_prim.shape[0] // 2
        x = self._merge_node_features(x_prim, x_ent, B)   # [B*(2 + N_ENTITY_TYPES*2), H]

        # 3. GCN message passing
        layer_outs = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = norm(x, batch)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_jumping_knowledge:
                layer_outs.append(x)

        # 4. Jumping Knowledge aggregation
        if self.use_jumping_knowledge:
            x = self.jk(layer_outs)   # [B*(2 + N_ENTITY_TYPES*2), H*num_layers]

        # 5. Global pooling: sum || mean || max  → [B, 3*dim]
        x_sum  = global_add_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x = torch.cat([x_sum, x_mean, x_max], dim=-1)   # [B, 3*dim]

        # 6. Deep readout MLP
        x = self.readout_mlp(x)   # [B, hidden] or [B, pool_out_dim] if layers=0

        # 7. Task-specific output
        return self.output_head(x)

    # ────────────────────────────────────────────────────────────────────
    def _merge_node_features(self, x_prim, x_ent, B):
        """
        Reconstruct the full [B*N_NODES, H] node feature tensor.

        Expected node layout per graph (matches graph construction):
            index 0                        → candidate  (primary)
            index 1                        → job desc   (primary)
            index 2 .. N_ENTITY_TYPES+1    → candidate entity nodes
            index N_ENTITY_TYPES+2 .. end  → job entity nodes
        """
        n_entity_nodes = N_ENTITY_TYPES * 2
        n_nodes        = 2 + n_entity_nodes   # primary + entity nodes per graph

        # x_prim : [B*2, H]              → [B, 2, H]
        # x_ent  : [B*n_entity_nodes, H] → [B, n_entity_nodes, H]
        xp = x_prim.view(B, 2, -1)
        xe = x_ent.view(B, n_entity_nodes, -1)

        x_all = torch.cat([xp, xe], dim=1)   # [B, n_nodes, H]

        return x_all.view(B * n_nodes, -1)   # [B*n_nodes, H]

    # ────────────────────────────────────────────────────────────────────
    def predict_proba(self, *args, **kwargs):
        """Convenience: returns probabilities instead of raw logits."""
        logits = self.forward(*args, **kwargs)
        return torch.sigmoid(logits)

