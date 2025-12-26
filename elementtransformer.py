import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import math
import processing
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class FVCOMDataset(Dataset):
    def __init__(
        self,
        node_data_dir: str,
        tri_data_dir: str,
        total_timesteps: int = 24 * 7,   # e.g., 7 days
        steps_per_file: int = 24,        # 1 file = 1 day = 24 steps
        t_in: int = 6,
        t_out: int = 6,
    ):
        self.node_data_dir = node_data_dir
        self.tri_data_dir = tri_data_dir
        self.steps_per_file = steps_per_file
        self.t_in = t_in
        self.t_out = t_out
        self.total_timesteps = total_timesteps

        assert steps_per_file % t_in == 0, f"steps_per_file ({steps_per_file}) must be divisible by t_in ({t_in})"
        assert steps_per_file % t_out == 0, f"steps_per_file ({steps_per_file}) must be divisible by t_out ({t_out})"
        assert total_timesteps % t_in == 0, f"total_timesteps ({total_timesteps}) must be divisible by t_in ({t_in})"

        self.node_files = sorted([f for f in os.listdir(node_data_dir) if f.endswith('.npy')])
        self.tri_files = sorted([f for f in os.listdir(tri_data_dir) if f.endswith('.npy')])
        assert len(self.node_files) == len(self.tri_files), "Node and triangle file counts must match!"

        self.num_blocks = total_timesteps // t_in
        self.total_samples = self.num_blocks - 1  # last block has no target

        if self.total_samples <= 0:
            raise ValueError(f"Not enough data: need at least {t_in * 2} timesteps")

    def _global_to_local(self, global_t: int):
        return global_t // self.steps_per_file, global_t % self.steps_per_file

    def _load_sequence(self, data_dir: str, files: list, start_t: int, length: int):
        """
        Load sequence assuming it lies entirely within one file.
        Only valid when blocks are aligned with file boundaries.
        """
        file_idx, local_t = self._global_to_local(start_t)
        if local_t + length > self.steps_per_file:
            raise RuntimeError(
                f"Sequence [t={start_t}, t+{length}) crosses file boundary. "
                "This should not happen if steps_per_file is divisible by t_in/t_out."
            )
        path = os.path.join(data_dir, files[file_idx])
        data = np.load(path)
        return data[local_t : local_t + length]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int):
        input_start = idx * self.t_in
        target_start = input_start + self.t_in

        node_in = self._load_sequence(self.node_data_dir, self.node_files, input_start, self.t_in)
        tri_in = self._load_sequence(self.tri_data_dir, self.tri_files, input_start, self.t_in)

        node_out = self._load_sequence(self.node_data_dir, self.node_files, target_start, self.t_out)
        tri_out = self._load_sequence(self.tri_data_dir, self.tri_files, target_start, self.t_out)

        indices1 = [0, 5, 10, 11, 12]
        indices2 = [0, 5, 10, 15, 16, 17]

        return (
            torch.from_numpy(node_in[:, :, indices1]).float(),
            torch.from_numpy(tri_in[:, :, indices2]).float()
        ), (
            torch.from_numpy(node_out[:, :, indices1]).float(),
            torch.from_numpy(tri_out[:, :, indices2]).float()
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
        channel_weights = weight_list
        self.register_buffer('channel_weights', channel_weights)

    def forward(self, pred, target):
        weights = self.channel_weights.view([1] * (pred.dim() - 1) + [-1])
        abs_error = torch.abs(pred - target)
        squared_error = (pred - target) ** 2
        weighted_abs_error = weights * abs_error
        weighted_squared_error = weights * squared_error
        mae = weighted_abs_error.mean()
        mse = weighted_squared_error.mean()
        loss = self.weight_mae * mae + self.weight_mse * mse
        return loss

class NodeSparseSelfAttention(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.q = nn.Linear(in_channels, out_channels)
        self.k = nn.Linear(in_channels, out_channels)
        self.v = nn.Linear(in_channels, out_channels)
        self.scale = out_channels ** -0.5  # 注意：通常是除以 sqrt(d)，所以是负指数！

    def forward(self, x, edge_index, size=None):
        """
        x: [t, N, in_channels]
        edge_index: [2, E]  (shared across all t time steps)
        returns: [t, N, out_channels]
        """
        t, N, _ = x.shape

        x_flat = x.view(t * N, -1)

        row, col = edge_index  # each of shape [E]
        offsets = torch.arange(t, device=x.device).view(-1, 1) * N  # [t, 1]
        row_exp = (row.unsqueeze(0) + offsets).view(-1)            # [t * E]
        col_exp = (col.unsqueeze(0) + offsets).view(-1)            # [t * E]
        edge_index_exp = torch.stack([row_exp, col_exp], dim=0)    # [2, t*E]

        q = self.q(x_flat)  # [t*N, out_channels]
        k = self.k(x_flat)  # [t*N, out_channels]
        v = self.v(x_flat)  # [t*N, out_channels]

        out_flat = self.propagate(edge_index_exp, q=q, k=k, v=v, size=(t * N, t * N))

        out = out_flat.view(t, N, -1)
        return out

    def message(self, q_i, k_j, v_j, index, ptr, size_i):
        # q_i, k_j: [num_edges, out_channels]
        attn_logits = (q_i * k_j).sum(dim=-1) * self.scale  # [E_total]
        alpha = softmax(attn_logits, index, num_nodes=size_i)  # [E_total]
        return alpha.unsqueeze(-1) * v_j  # [E_total, out_channels]

    def update(self, aggr_out, x=None):
        return aggr_out

class TriangleSparseSelfAttention(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.q = nn.Linear(in_channels, out_channels)
        self.k = nn.Linear(in_channels, out_channels)
        self.v = nn.Linear(in_channels, out_channels)
        self.scale = out_channels ** 0.5

    def forward(self, x, edge_index, size=None):
        """
        x: [t * M, in_channels]  <-- reshape to node-first
        or keep [t, M, C] but use vmap / broadcasting carefully
        But easier: treat as [N, C] where N = t * M, and adjust edge_index accordingly.
        """
        t, M, _ = x.shape
        N = t * M

        # Reshape x to [N, C]
        x_flat = x.view(N, -1)

        # Expand edge_index for each time step (assuming same graph per time step)
        # edge_index: [2, E] -> [2, t * E]
        row, col = edge_index
        offsets = torch.arange(t, device=x.device).view(-1, 1) * M  # [t, 1]
        row_exp = (row.unsqueeze(0) + offsets).view(-1)            # [t * E]
        col_exp = (col.unsqueeze(0) + offsets).view(-1)            # [t * E]
        edge_index_exp = torch.stack([row_exp, col_exp], dim=0)    # [2, t*E]

        # Compute Q, K, V on flat input
        q = self.q(x_flat)  # [N, out_channels]
        k = self.k(x_flat)  # [N, out_channels]
        v = self.v(x_flat)  # [N, out_channels]

        # Propagate
        out_flat = self.propagate(edge_index_exp, q=q, k=k, v=v, size=(N, N))
        out = out_flat.view(t, M, -1)
        return out

    def message(self, q_i, k_j, v_j, index, ptr, size_i):
        # q_i: target query, k_j/v_j: source key/value
        attn_logits = (q_i * k_j).sum(dim=-1) * self.scale  # [E]
        alpha = softmax(attn_logits, index, num_nodes=size_i)  # [E]
        return alpha.unsqueeze(-1) * v_j  # [E, out_channels]

    def update(self, aggr_out, x):
        return aggr_out

class NodeToTriangleCrossAttention(MessagePassing):
    def __init__(self, node_dim, tri_dim, hidden_dim, dropout=0.0):
        super().__init__(aggr='add')
        self.q = nn.Linear(tri_dim, hidden_dim)      # Query from triangle (target)
        self.k = nn.Linear(node_dim, hidden_dim)     # Key from node (source)
        self.v = nn.Linear(node_dim, hidden_dim)     # Value from node (source)
        self.out_proj = nn.Linear(hidden_dim, tri_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, x_tri, x_node, nt_edge_index, size=None, return_attn=False):
        """
        x_tri: [t, M, tri_dim]
        x_node: [t, N, node_dim]
        nt_edge_index: [2, E], source=node, target=triangle (static across t)
        """
        t, M, _ = x_tri.shape
        _, N, _ = x_node.shape

        # Flatten: [t, M/N, C] -> [t*M/t*N, C]
        x_tri_flat = x_tri.view(t * M, -1)   # [tM, tri_dim]
        x_node_flat = x_node.view(t * N, -1) # [tN, node_dim]

        # Expand edge_index: shift node and triangle indices by time step
        src, dst = nt_edge_index  # src: node idx (0~N-1), dst: tri idx (0~M-1)
        offsets_t = torch.arange(t, device=x_tri.device).view(-1, 1)  # [t, 1]

        # Source (node): add i * N
        src_exp = (src.unsqueeze(0) + offsets_t * N).view(-1)   # [t*E]
        # Target (triangle): add i * M
        dst_exp = (dst.unsqueeze(0) + offsets_t * M).view(-1)   # [t*E]

        edge_index_exp = torch.stack([dst_exp, src_exp], dim=0)

        edge_index_exp = torch.stack([src_exp, dst_exp], dim=0)  # [2, t*E], [source=node, target=tri]

        # Compute Q/K/V
        q = self.q(x_tri_flat)    # [tM, hidden]
        k = self.k(x_node_flat)   # [tN, hidden]
        v = self.v(x_node_flat)   # [tN, hidden]

        # Propagate
        aggr_out = self.propagate(edge_index_exp, q=q, k=k, v=v, size=size)

        # Reshape and output projection
        aggr_out = aggr_out.view(t, M, -1)  # [t, M, hidden]
        out = self.out_proj(aggr_out) + x_tri  # residual

        if return_attn:
            raise NotImplementedError("return_attn not supported in batched version yet.")
        else:
            return out

    def message(self, q_i, k_j, v_j, index, ptr, size_i):
        attn_logits = (q_i * k_j).sum(dim=-1) * self.scale
        alpha = softmax(attn_logits, index, num_nodes=size_i)
        return alpha.unsqueeze(-1) * v_j

    def update(self, aggr_out):
        return aggr_out

class TriangleToNodeCrossAttention(MessagePassing):
    def __init__(self, tri_dim, node_dim, hidden_dim, dropout=0.0):
        super().__init__(aggr='add')
        self.q = nn.Linear(node_dim, hidden_dim)      # Query from node (target)
        self.k = nn.Linear(tri_dim, hidden_dim)       # Key from triangle (source)
        self.v = nn.Linear(tri_dim, hidden_dim)       # Value from triangle (source)
        self.out_proj = nn.Linear(hidden_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, x_node, x_tri, tn_edge_index, size=None, return_attn=False):
        """
        x_node: [t, N, node_dim]
        x_tri:  [t, M, tri_dim]
        tn_edge_index: [2, E], source=triangle, target=node (static across t)
        """
        
        t, N, _ = x_node.shape
        _, M, _ = x_tri.shape

        x_node_flat = x_node.view(t * N, -1)  # [tN, node_dim]
        x_tri_flat = x_tri.view(t * M, -1)    # [tM, tri_dim]

        src, dst = tn_edge_index  # src: tri idx (0~M-1), dst: node idx (0~N-1)
        offsets_t = torch.arange(t, device=x_node.device).view(-1, 1)

        # Source (triangle): + i * M
        src_exp = (src.unsqueeze(0) + offsets_t * M).view(-1)   # [t*E]
        # Target (node): + i * N
        dst_exp = (dst.unsqueeze(0) + offsets_t * N).view(-1)   # [t*E]

        # edge_index = [source=tri, target=node]
        edge_index_exp = torch.stack([src_exp, dst_exp], dim=0)  # [2, t*E]

        q = self.q(x_node_flat)   # [tN, hidden]
        k = self.k(x_tri_flat)    # [tM, hidden]
        v = self.v(x_tri_flat)    # [tM, hidden]

        aggr_out = self.propagate(edge_index_exp, q=q, k=k, v=v, size=size)
        aggr_out = aggr_out.view(t, N, -1)
        out = self.out_proj(aggr_out) + x_node

        if return_attn:
            raise NotImplementedError("return_attn not supported in batched version yet.")
        else:
            return out

    def message(self, q_i, k_j, v_j, index, ptr, size_i):
        attn_logits = (q_i * k_j).sum(dim=-1) * self.scale
        alpha = softmax(attn_logits, index, num_nodes=size_i)
        return alpha.unsqueeze(-1) * v_j

    def update(self, aggr_out):
        return aggr_out

class CrossAttentionTransformer(nn.Module):
    def __init__(self, embed_dim=256, nhead=1, dropout=0.1, mlp_ratio=4, t_in=1, node=60882, triangle=115443,):
        super().__init__()
        self.num_node_nodes = node
        self.num_tri_nodes = triangle
        self.t_in = t_in
        node_edge_index, tri_edge_index, tn_edge_index, nt_edge_index = processing.generate_sparse_graph()
        self.register_buffer('node_edge_index', node_edge_index)
        self.register_buffer('tri_edge_index', tri_edge_index)
        self.register_buffer('tn_edge_index', tn_edge_index)
        self.register_buffer('nt_edge_index', nt_edge_index)

        self.embed_dim = embed_dim
        self.nhead = nhead
        self.mlp_ratio = mlp_ratio
        self.dropout = nn.Dropout(dropout)

        self.node_self_attn  = NodeSparseSelfAttention(self.embed_dim, self.embed_dim)
        self.node2tri = NodeToTriangleCrossAttention(node_dim=self.embed_dim, tri_dim=self.embed_dim, hidden_dim=self.embed_dim, dropout=0.1)
        self.triangle_self_attn = TriangleSparseSelfAttention(self.embed_dim, self.embed_dim)
        self.tri2node = TriangleToNodeCrossAttention(node_dim=self.embed_dim, tri_dim=self.embed_dim, hidden_dim=self.embed_dim, dropout=0.1)
    
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

        node_self_attn = self.node_self_attn(self.norm_node_self(node_original),
                                             self.node_edge_index,
                                             size=(self.num_node_nodes, self.num_node_nodes))
        node = node_original + self.dropout(node_self_attn)
        self.x_node_new = self.tri2node(self.norm_node_cross(node),
                                        self.norm_triangle_self(triangle_original),
                                        self.tn_edge_index,
                                        size=(self.t_in * self.num_tri_nodes, self.t_in * self.num_node_nodes),
                                        return_attn=False)
        
        node = node + self.dropout(self.x_node_new)
        node = node + self.dropout(self.mlp_node(self.norm_node_mlp(node)))

        triangle_self_attn = self.triangle_self_attn(self.norm_triangle_self(triangle_original),
                                                     self.tri_edge_index,
                                                     size=(self.num_tri_nodes, self.num_tri_nodes))
        
        triangle = triangle_original + self.dropout(triangle_self_attn)
        self.x_tri_new  = self.node2tri(self.norm_triangle_cross(triangle),
                                        self.norm_node_self(node_original),
                                        self.nt_edge_index,
                                        size=(self.t_in * self.num_node_nodes, self.t_in * self.num_tri_nodes),
                                        return_attn=False)
        triangle = triangle + self.dropout(self.x_tri_new)
        triangle = triangle + self.dropout(self.mlp_triangle(self.norm_triangle_mlp(triangle)))

        return node, triangle

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
        self.layers = nn.ModuleList([
            CrossAttentionTransformer(self.embed_dim, self.nhead, dropout, self.mlp_ratio)
            for _ in range(self.num_layers)
        ])
        self._init_weights()

    def _init_weights(self):
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
        # print('-1', node.shape, triangle.shape)
        T_node, N_node, C_node = node.shape
        T_triangle, N_triangle, C_triangle = triangle.shape
        assert N_node == self.node, f"Expected {self.node} nodes, got {N_node}"
        assert N_triangle == self.triangle, f"Expected {self.triangle} triangles, got {N_triangle}"
        assert C_node == self.node_var, f"Expected {self.node_var} node features, got {C_node}"
        assert C_triangle == self.triangle_var, f"Expected {self.triangle_var} triangle features, got {C_triangle}"
        # print('0', node.shape, triangle.shape)
        node = self.node_embedding_layer(node)
        triangle = self.triangle_embedding_layer(triangle)
        # print('1', node.shape, triangle.shape)
        for layer in self.layers:
            node, triangle = layer(node, triangle)
            # print('2', node.shape, triangle.shape)

        # print('3', node.shape, triangle.shape)
        return node, triangle

class MultiStepPredictionHead(nn.Module):
    def __init__(self, embed_dim, t_in, t_out, out_dim, dropout=0.1):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.out_dim = out_dim
        
        self.head = nn.Sequential(
            nn.LayerNorm(t_in * embed_dim),
            nn.Linear(t_in * embed_dim, t_in * embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(t_in * embed_dim // 2, t_out * out_dim)
        )

    def forward(self, x):
        # x: [t_in, N, C]
        t, n, c = x.shape
        assert t == self.t_in, f"Expected t={self.t_in}, got {t}"
        
        x = x.permute(1, 0, 2).reshape(n, -1)  # [N, t_in * C]
        # print('x', x.shape)
        out = self.head(x)                    # [N, t_out * out_dim]
        out = out.reshape(n, self.t_out, self.out_dim).permute(1, 0, 2)  # [t_out, N, out_dim]
        return out

class Decoder(nn.Module):
    def __init__(self, node_var=11, triangle_var=18, embed_dim=256, t_in=6, t_out=6, dropout=0.1):
        super().__init__()
        # self.node_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim // 2, node_var)
        # )
        
        # self.triangle_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim // 2, triangle_var)
        # )

        self.node_pred_head = MultiStepPredictionHead(
            embed_dim=embed_dim,
            t_in=t_in,
            t_out=t_out,
            out_dim=node_var,
            dropout=dropout
        )

        self.tri_pred_head = MultiStepPredictionHead(
            embed_dim=embed_dim,
            t_in=t_in,
            t_out=t_out,
            out_dim=triangle_var,
            dropout=dropout
        )

    def forward(self, node, triangle):
        # print('4', node.shape, triangle.shape)
        node_pred = self.node_pred_head(node)      # [t_out, N, node_var]
        triangle_pred = self.tri_pred_head(triangle)         # [t_out, M, triangle_var]

        # node_pred = self.node_head(node)
        # triangle_pred = self.triangle_head(triangle)
        # print('5', node.shape, triangle.shape)
        return node_pred, triangle_pred

class ElementTransformerNet(nn.Module):
    def __init__(self, node=60882, triangle=115443, node_var=13,
                 triangle_var=18, embed_dim=256,
                 mlp_ratio=4., nhead=2, num_layers=2,
                 t_in=6, t_out=6,
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
            t_in=t_in,
            t_out=t_out,
            triangle_var=triangle_var
        )

    def forward(self, node, triangle):
        node, triangle = self.encoder(node, triangle)
        node, triangle = self.decoder(node, triangle)
        return node, triangle
    
    def predict(self, node_input_data, triangle_input_data, checkpoint_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_name, map_location=device)

        self.load_state_dict(checkpoint['model_state_dict'])

        self.eval()
        with torch.no_grad():
            output = self(node_input_data, triangle_input_data)

        # prediction = output.squeeze(0).cpu().numpy()
        
        return output

def FVCOMModel(node=60882, triangle=115443, node_var=13,
               triangle_var=18, embed_dim=256,
               mlp_ratio=4., nhead=2, num_layers=2,
               t_in=6, t_out=6,
               neighbor_table=None, dropout=0.1):
    
    model = ElementTransformerNet(node=node, triangle=triangle, node_var=node_var,
                                  triangle_var=triangle_var, embed_dim=embed_dim,
                                  mlp_ratio=mlp_ratio, nhead=nhead, num_layers=num_layers,
                                  t_in=t_in, t_out=t_out,
                                  neighbor_table=neighbor_table, dropout=dropout)
    return model

if __name__ == "__main__":
    pass
    