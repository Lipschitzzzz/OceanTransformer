import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import math
import processing
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import json

class FVCOMDataset(Dataset):
    def __init__(
        self,
        node_data_dir: str,
        tri_data_dir: str,
        total_timesteps: int = 144 * 7,
        steps_per_file: int = 144,
        pred_step: int = 1
    ):
        self.node_data_dir = node_data_dir
        self.tri_data_dir = tri_data_dir
        self.steps_per_file = steps_per_file
        self.pred_step = pred_step

        # Get sorted .npy files (assume aligned naming: node_*.npy ↔ tri_*.npy)
        self.node_files = sorted([f for f in os.listdir(node_data_dir) if f.endswith('.npy')])
        self.tri_files = sorted([f for f in os.listdir(tri_data_dir) if f.endswith('.npy')])

        assert len(self.node_files) == len(self.tri_files), "Number of node and triangle files must match!"
        
        self.total_timesteps = total_timesteps
        self.max_start_t = total_timesteps - pred_step - 1
        if self.max_start_t < 0:
            raise ValueError(f"pred_step={pred_step} too large for total_timesteps={total_timesteps}")
        self.total_samples = self.max_start_t + 1

    def _global_to_local(self, global_t: int):
        """Convert global timestep to (file_index, local_timestep)."""
        file_idx = global_t // self.steps_per_file
        local_t = global_t % self.steps_per_file
        return file_idx, local_t

    def _load_sequence(self, data_dir: str, files: list, global_t: int, length: int):
        """Load a sequence of `length` steps starting at `global_t`."""
        file_idx, local_t = self._global_to_local(global_t)
        if local_t + length > self.steps_per_file:
            raise RuntimeError(
                f"Sample crosses file boundary at t={global_t}. "
                "Either reduce pred_step or implement cross-file loading."
            )
        path = os.path.join(data_dir, files[file_idx])
        data = np.load(path)  # shape: [T=144, N, C]
        seq = data[local_t : local_t + length]  # [length, N, C]
        return seq

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int):
        t_input = idx
        t_target = idx + self.pred_step

        # Load input sequences (for encoder / history)
        node_input = self._load_sequence(self.node_data_dir, self.node_files, t_input, self.pred_step).squeeze(0)
        tri_input = self._load_sequence(self.tri_data_dir, self.tri_files, t_input, self.pred_step).squeeze(0)

        # Load single-step targets
        node_target = self._load_sequence(self.node_data_dir, self.node_files, t_target, 1).squeeze(0)
        tri_target = self._load_sequence(self.tri_data_dir, self.tri_files, t_target, 1).squeeze(0)

        # print('1: ', node_input.shape)
        # print('2: ', tri_input.shape)
        # print('3: ', node_target.shape)
        # print('4: ', tri_target.shape)

        return (
            torch.from_numpy(node_input).float(),
            torch.from_numpy(tri_input).float()
        ), (
            torch.from_numpy(node_target).float(),
            torch.from_numpy(tri_target).float()
        )

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding from "Attention Is All You Need".
    
    Shape:
        Input:  (batch_size, seq_len, d_model)
        Output: (batch_size, seq_len, d_model)  [input + positional encoding]
    
    Args:
        d_model (int): Embedding dimension.
        max_len (int, optional): Maximum sequence length for pre-computation (for efficiency).
                                 Even if actual seq_len > max_len, it will compute on-the-fly.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Precompute positional encodings for positions [0, max_len)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model // 2,)
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        
        # Register as buffer (not a model parameter, but saved with state_dict)
        self.register_buffer('pe', pe, persistent=False)  # persistent=False to avoid saving if not needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            x + positional_encoding (same shape)
        """
        seq_len = x.size(1)
        
        # If sequence is longer than precomputed max_len, compute dynamically
        if seq_len > self.pe.size(0):
            # Recompute for the required length
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.d_model, 2, dtype=torch.float, device=x.device) 
                * (-math.log(10000.0) / self.d_model)
            )
            pe_dynamic = torch.zeros(seq_len, self.d_model, device=x.device)
            pe_dynamic[:, 0::2] = torch.sin(position * div_term)
            pe_dynamic[:, 1::2] = torch.cos(position * div_term)
            pe_to_use = pe_dynamic
        else:
            pe_to_use = self.pe[:seq_len]

        # Add positional encoding (broadcast over batch)
        return x + pe_to_use.unsqueeze(0)

class WeightedMAEMSELoss(nn.Module):
    def __init__(self, weight_mae=1.0, weight_mse=0.2, var=10, weight_list = torch.ones(1)):
        super().__init__()
        self.weight_mae = weight_mae
        self.weight_mse = weight_mse
        self.var = var

        channel_weights = torch.ones(var)
        # print(channel_weights)
        channel_weights = weight_list
        self.register_buffer('channel_weights', channel_weights)

    def forward(self, pred, target):
        weights = self.channel_weights.view([1] * (pred.dim() - 1) + [-1])
        # print(pred.shape)
        # print(target.shape)
        # print(weights.shape)
        abs_error = torch.abs(pred - target)
        squared_error = (pred - target) ** 2
        weighted_abs_error = weights * abs_error
        weighted_squared_error = weights * squared_error
        mae = weighted_abs_error.mean()
        mse = weighted_squared_error.mean()
        loss = self.weight_mae * mae + self.weight_mse * mse
        return loss
    
class NodeEmbedding(nn.Module):
    def __init__(self, var_in=11, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(var_in, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)
    
class TriangleEmbedding(nn.Module):
    def __init__(self, var_in=18, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(var_in, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    def forward(self, x):
        return self.net(x)

class NodeSparseSelfAttention(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        self.scale = out_channels ** 0.5

    def forward(self, x, edge_index):
        # x: [nnode, in_channels]
        # edge_index: [2, E]
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, index):
        # x_i: source (target node), x_j: neighbor (source node)
        Q = self.q_proj(x_i)          # [E, out]
        K = self.k_proj(x_j)          # [E, out]
        V = self.v_proj(x_j)          # [E, out]
        attn_logits = (Q * K).sum(dim=-1) / self.scale  # [E]
        attn_logits = attn_logits.flatten()
        attn_weights = softmax(attn_logits, index)      # 归一化到每个节点的邻居
        return attn_weights.unsqueeze(-1) * V           # [E, out]

    def update(self, aggr_out, x):
        return aggr_out  # [nnode, out_channels]

class TriangleSparseSelfAttention(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        self.scale = out_channels ** 0.5

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, index):
        Q = self.q_proj(x_i)
        K = self.k_proj(x_j)
        V = self.v_proj(x_j)
        
        attn_logits = (Q * K).sum(dim=-1) / self.scale
        attn_logits = attn_logits.flatten()
        attn_weights = softmax(attn_logits, index)
        return attn_weights.unsqueeze(-1) * V

    def update(self, aggr_out, x):
        return aggr_out
    
class NodeToTriangleCrossAttention(MessagePassing):
    def __init__(self, node_dim, tri_dim, hidden_dim, dropout=0.0):
        super().__init__(aggr='add')
        self.q = nn.Linear(tri_dim, hidden_dim)
        self.k = nn.Linear(node_dim, hidden_dim)
        self.v = nn.Linear(node_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, tri_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, x_tri, x_node, nt_edge_index, return_attn=False):
        return self.propagate(nt_edge_index, x=(x_node, x_tri), return_attn=return_attn)

    def message(self, x_i, x_j, index, return_attn):
        q = self.q(x_j); k = self.k(x_i); v = self.v(x_i)
        alpha = (q * k).sum(dim=-1) * self.scale
        alpha = softmax(alpha, index)
        alpha = self.dropout(alpha)
        if return_attn: self._alpha = alpha
        return alpha.unsqueeze(-1) * v
    def update(self, aggr_out, x, return_attn):
        x_dst = x[1]  # ← x_dst is x_tri (triangle features)
        out = self.out(aggr_out) + x_dst
        return (out, self._alpha) if return_attn else out
        
class TriangleToNodeCrossAttention(MessagePassing):
    def __init__(self, tri_dim, node_dim, hidden_dim, dropout=0.0):
        super().__init__(aggr='add')
        self.q = nn.Linear(node_dim, hidden_dim)
        self.k = nn.Linear(tri_dim, hidden_dim)
        self.v = nn.Linear(tri_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, x_node, x_tri, tn_edge_index, return_attn=False):
        assert tn_edge_index[1].max() < x_node.shape[0], "Node index out of bounds!"
        assert tn_edge_index[0].max() < x_tri.shape[0], "Triangle index out of bounds!"
        return self.propagate(tn_edge_index, x=(x_tri, x_node), return_attn=return_attn)

    def message(self, x_i, x_j, index, return_attn):
        q = self.q(x_j); k = self.k(x_i); v = self.v(x_i)
        alpha = (q * k).sum(dim=-1) * self.scale
        alpha = softmax(alpha, index)
        alpha = self.dropout(alpha)
        if return_attn: self._alpha = alpha
        return alpha.unsqueeze(-1) * v

    def update(self, aggr_out, x, return_attn):
        x_dst = x[1]  # ← x_dst is x_tri (triangle features)
        out = self.out(aggr_out) + x_dst
        return (out, self._alpha) if return_attn else out

class CrossAttentionTransformer(nn.Module):
    def __init__(self, embed_dim=256, nhead=1, dropout=0.1, mlp_ratio=4):
        super().__init__()
        node_edge_index, tri_edge_index, tn_edge_index, nt_edge_index = processing.generate_sparse_graph()
        self.register_buffer('node_edge_index', node_edge_index)
        self.register_buffer('tri_edge_index', tri_edge_index)
        self.register_buffer('tn_edge_index', tn_edge_index)
        self.register_buffer('nt_edge_index', nt_edge_index)
        # print("nn_edge_index shape:", node_edge_index.shape)
        # print("tt_edge_index shape:", tri_edge_index.shape)
        # print("tn_edge_index shape:", tn_edge_index.shape, np.max(self.tn_edge_index))
        # print("nt_edge_index shape:", nt_edge_index.shape, np.max(self.tn_edge_index))

        self.embed_dim = embed_dim
        self.nhead = nhead
        self.mlp_ratio = mlp_ratio
        self.dropout = nn.Dropout(dropout)

        self.node_self_attn  = NodeSparseSelfAttention(self.embed_dim, self.embed_dim)
        self.node2tri = NodeToTriangleCrossAttention(node_dim=self.embed_dim, tri_dim=self.embed_dim, hidden_dim=self.embed_dim, dropout=0.1)
        # self.node_self_attn = nn.MultiheadAttention(self.embed_dim, self.nhead, batch_first=True, dropout=dropout)
        self.triangle_self_attn = TriangleSparseSelfAttention(self.embed_dim, self.embed_dim)
        self.tri2node = TriangleToNodeCrossAttention(node_dim=self.embed_dim, tri_dim=self.embed_dim, hidden_dim=self.embed_dim, dropout=0.1)
        # self.triangle_self_attn = nn.MultiheadAttention(self.embed_dim, self.nhead, batch_first=True, dropout=dropout)

        self.node_to_triangle_attn = nn.MultiheadAttention(
            self.embed_dim, self.nhead, batch_first=True, dropout=dropout
        )

        self.triangle_to_node_attn = nn.MultiheadAttention(
            self.embed_dim, self.nhead, batch_first=True, dropout=dropout
        )
    
        self.norm_node_self = nn.LayerNorm(embed_dim)  # node self
        self.norm_triangle_self = nn.LayerNorm(embed_dim)  # triangle self
        self.norm_node_cross = nn.LayerNorm(embed_dim)  # node cross
        self.norm_triangle_cross = nn.LayerNorm(embed_dim)  # triangle cross
        self.norm_node_mlp = nn.LayerNorm(embed_dim)  # node mlp
        self.norm_triangle_mlp = nn.LayerNorm(embed_dim)  # triangle mlp

        self.mlp_node = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * self.mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.embed_dim * self.mlp_ratio), self.embed_dim),
            nn.Dropout(dropout)
        )

        self.mlp_triangle = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * self.mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.embed_dim * self.mlp_ratio), self.embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, node, triangle):
        node_original = node
        triangle_original = triangle

        node_self_attn = self.node_self_attn(self.norm_node_self(node_original),self.node_edge_index)
        node = node_original + self.dropout(node_self_attn)
        self.x_node_new = self.tri2node(self.norm_node_cross(node),
                                        self.norm_triangle_self(triangle_original),
                                        self.tn_edge_index)
        # node_cross_attn, _ = self.node_to_triangle_attn(
        #     self.norm_node_cross(node),
        #     self.norm_triangle_self(triangle_original),
        #     self.norm_triangle_self(triangle_original)
        # )
        node = node + self.dropout(self.x_node_new)
        node = node + self.dropout(self.mlp_node(self.norm_node_mlp(node)))

        triangle_self_attn = self.triangle_self_attn(self.norm_triangle_self(triangle_original),self.tri_edge_index)
        triangle = triangle_original + self.dropout(triangle_self_attn)
        # triangle_cross_attn, _ = self.triangle_to_node_attn(
        #     self.norm_triangle_cross(triangle),
        #     self.norm_node_self(node_original),
        #     self.norm_node_self(node_original)
        # )
        self.x_tri_new  = self.node2tri(self.norm_triangle_cross(triangle),
                                        self.norm_node_self(node_original), self.nt_edge_index)
        triangle = triangle + self.dropout(self.x_tri_new)
        triangle = triangle + self.dropout(self.mlp_triangle(self.norm_triangle_mlp(triangle)))

        return node, triangle

class NeighborSparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, neighbor_table, dropout=0.1):
        """
        neighbor_table: numpy array or torch tensor of shape (K, N)
                        where K=3, N=seq_len.
                        Each column i lists up to K neighbor indices for token i.
                        Use -1 to indicate "no neighbor".
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        if isinstance(neighbor_table, np.ndarray):
            neighbor_table = torch.from_numpy(neighbor_table).long()
        else:
            neighbor_table = neighbor_table.long()

        K, N = neighbor_table.shape
        self.K = K
        self.N = N

        valid_mask = neighbor_table != -1  # (K, N), bool
        neighbor_table = neighbor_table.clamp(min=0)

        self.register_buffer("neighbor_indices", neighbor_table)
        self.register_buffer("valid_mask", valid_mask.float())

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.N, f"Input sequence length {N} != expected {self.N}"

        # Project Q, K, V
        q = self.q_proj(x)  # (B, N, C)
        k = self.k_proj(x)  # (B, N, C)
        v = self.v_proj(x)  # (B, N, C)

        # Reshape for multi-head: (B, N, H, D) -> (B, H, N, D)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, D)

        # Gather neighbors for K and V: for each position i, get its K neighbors
        # neighbor_indices: (K, N) -> expand to (B, H, K, N)
        idx = self.neighbor_indices.unsqueeze(0).unsqueeze(0)  # (1, 1, K, N)
        idx = idx.expand(B, self.num_heads, -1, -1)  # (B, H, K, N)

        # k: (B, H, N, D) -> gather over N dimension using idx -> (B, H, K, N, D)
        k_neighbors = torch.gather(k.unsqueeze(2).expand(-1, -1, self.K, -1, -1), 3, idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim))
        v_neighbors = torch.gather(v.unsqueeze(2).expand(-1, -1, self.K, -1, -1), 3, idx.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim))

        # q for each position: (B, H, N, D) -> unsqueeze to (B, H, 1, N, D)
        q_exp = q.unsqueeze(2)  # (B, H, 1, N, D)

        # Compute attention scores: (B, H, K, N)
        attn_scores = (q_exp * k_neighbors).sum(dim=-1) * self.scale  # (B, H, K, N)

        # Apply validity mask: invalid neighbors get -inf
        valid_mask = self.valid_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, K, N)
        attn_scores = attn_scores.masked_fill(valid_mask == 0, float('-inf'))

        # Softmax over K dimension (only among valid neighbors)
        attn_weights = attn_scores.softmax(dim=2)  # (B, H, K, N)
        attn_weights = self.attn_drop(attn_weights)

        # Weighted sum of V neighbors
        out = (attn_weights.unsqueeze(-1) * v_neighbors).sum(dim=2)  # (B, H, N, D)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out
    
class SparseTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, neighbor_table, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = NeighborSparseAttention(embed_dim, num_heads, neighbor_table, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, node=60882, triangle=115443, node_var=11,
                 triangle_var=18, embed_dim=256,
                 mlp_ratio=4., nhead=2, num_layers=2,
                 neighbor_table=None, dropout=0.1):
        super().__init__()
        self.node = node
        self.triangle = triangle
        self.embed_dim = embed_dim
        self.node_var = node_var
        self.triangle_var = triangle_var
        self.mlp_ratio = mlp_ratio
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.node_embedding_layer = nn.Linear(self.node_var, self.embed_dim)
        self.triangle_embedding_layer = nn.Linear(self.triangle_var, self.embed_dim)
        if neighbor_table != None:
            assert neighbor_table.shape == (3, self.triangle), f"neighbor_table must be (3, {self.triangle})"
        # self.pos_drop = nn.Dropout(p=dropout)
        # self.transformer_blocks = nn.ModuleList([
        #     SparseTransformerBlock(
        #         embed_dim=embed_dim,
        #         num_heads=num_heads,
        #         mlp_ratio=mlp_ratio,
        #         neighbor_table=neighbor_table,
        #         dropout=dropout
        #     ) for _ in range(num_layers)
        # ])
        self.layers = nn.ModuleList([
            CrossAttentionTransformer(self.embed_dim, self.nhead, dropout, self.mlp_ratio)
            for _ in range(self.num_layers)
        ])

        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.embed_dim,
        #     nhead=self.nhead,
        #     dim_feedforward=int(self.embed_dim * self.mlp_ratio),
        #     dropout=self.dropout,
        #     activation='gelu',
        #     batch_first=True
        # )
        # self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self._init_weights()

    def _init_weights(self):
        # nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, node, triangle):
        node = node.squeeze(0)
        triangle = triangle.squeeze(0)
        N_node, C_node = node.shape
        N_triangle, C_triangle = triangle.shape
        assert N_node == self.node, f"Expected {self.node} nodes, got {N_node}"
        assert N_triangle == self.triangle, f"Expected {self.triangle} triangles, got {N_triangle}"
        assert C_node == self.node_var, f"Expected {self.node_var} node features, got {C_node}"
        assert C_triangle == self.triangle_var, f"Expected {self.triangle_var} triangle features, got {C_triangle}"
        node = self.node_embedding_layer(node)
        triangle = self.triangle_embedding_layer(triangle)

        for layer in self.layers:
            node, triangle = layer(node, triangle)

        return node, triangle

class Decoder(nn.Module):
    def __init__(self, node_var=11, triangle_var=18, embed_dim=256, dropout=0.1):
        super().__init__()
        self.node_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, node_var)
        )
        
        self.triangle_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, triangle_var)
        )

    def forward(self, node, triangle):
        node_pred = self.node_head(node)
        triangle_pred = self.triangle_head(triangle)
        return node_pred, triangle_pred

class ElementTransformerNet(nn.Module):
    def __init__(self, node=60882, triangle=115443, node_var=11,
                 triangle_var=18, embed_dim=256,
                 mlp_ratio=4., nhead=2, num_layers=2,
                 neighbor_table=None, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(
            node_var=node_var,
            triangle_var=triangle_var,
            embed_dim=embed_dim,
            node=node,
            triangle=triangle,
            nhead=nhead,
            mlp_ratio=mlp_ratio,
            neighbor_table=neighbor_table,
            dropout=dropout,
            num_layers=num_layers
        )
        self.decoder = Decoder(
            embed_dim=embed_dim,
            node_var=node_var,
            triangle_var=triangle_var
        )

    def forward(self, node, triangle):
        node, triangle = self.encoder(node, triangle)
        node, triangle = self.decoder(node, triangle)
        return node, triangle
    
    def predict(self, checkpoint_name, input_data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_name, map_location=device)

        self.load_state_dict(checkpoint['model_state_dict'])

        self.eval()
        with torch.no_grad():
            output = self(input_data)

        # prediction = output.squeeze(0).cpu().numpy()
        
        return output

def FVCOMModel(node=60882, triangle=115443, node_var=13,
               triangle_var=18, embed_dim=256,
               mlp_ratio=4., nhead=2, num_layers=2,
               neighbor_table=None, dropout=0.1):
    
    model = ElementTransformerNet(node=node, triangle=triangle, node_var=node_var,
                                  triangle_var=triangle_var, embed_dim=embed_dim,
                                  mlp_ratio=mlp_ratio, nhead=nhead, num_layers=num_layers,
                                  neighbor_table=neighbor_table, dropout=dropout)
    return model

if __name__ == "__main__":
    pass
    