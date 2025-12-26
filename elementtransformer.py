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
        self.q = nn.Linear(in_channels, out_channels)
        self.k = nn.Linear(in_channels, out_channels)
        self.v = nn.Linear(in_channels, out_channels)
        self.scale = out_channels ** -0.5

    def forward(self, x, edge_index, size):
        """
        x: [N, in_channels]
        edge_index: [2, E] (static graph)
        returns: [N, out_channels]
        """
        q = self.q(x)  # [N, C]
        k = self.k(x)
        v = self.v(x)
        return self.propagate(edge_index, q=q, k=k, v=v, size=size)

    def message(self, q_i, k_j, v_j, index, size_i):
        attn_logits = (q_i * k_j).sum(dim=-1) * self.scale  # [E]
        alpha = softmax(attn_logits, index, num_nodes=size_i)
        return alpha.unsqueeze(-1) * v_j

    def update(self, aggr_out):
        return aggr_out


class TriangleSparseSelfAttention(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.q = nn.Linear(in_channels, out_channels)
        self.k = nn.Linear(in_channels, out_channels)
        self.v = nn.Linear(in_channels, out_channels)
        self.scale = out_channels ** -0.5

    def forward(self, x, edge_index, size):
        """
        x: [M, in_channels]
        edge_index: [2, E_tri]
        returns: [M, out_channels]
        """
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        return self.propagate(edge_index, q=q, k=k, v=v, size=size)

    def message(self, q_i, k_j, v_j, index, size_i):
        attn_logits = (q_i * k_j).sum(dim=-1) * self.scale
        alpha = softmax(attn_logits, index, num_nodes=size_i)
        return alpha.unsqueeze(-1) * v_j

    def update(self, aggr_out):
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

    def forward(self, x_tri, x_node, nt_edge_index, size, return_attn=False):
        """
        x_tri: [M, tri_dim]
        x_node: [N, node_dim]
        nt_edge_index: [2, E], source=node, target=triangle
        returns: [M, tri_dim]
        """
        if return_attn:
            raise NotImplementedError("return_attn not supported.")

        q = self.q(x_tri)    # [M, H]
        k = self.k(x_node)   # [N, H]
        v = self.v(x_node)   # [N, H]

        # Note: edge_index format in PyG: [target, source] for message passing?
        # But your original code uses [src, dst] = nt_edge_index with src=node, dst=tri
        # In propagate, we need: edge_index[0] = target (tri), edge_index[1] = source (node)
        # So we pass as [dst, src] = [tri_idx, node_idx]
        # But your original had: edge_index_exp = [src_exp, dst_exp] -> [node, tri] -> WRONG for MessagePassing!
        # Correction: MessagePassing expects edge_index[0] = target, edge_index[1] = source
        # So we should pass: [dst, src] = [tri, node]
        # However, your original code did: torch.stack([src_exp, dst_exp]) and used q_i (target), k_j (source)
        # That implies you treated edge_index as [source, target] — which is non-standard.
        #
        # To stay consistent with your original logic where:
        #   q_i = query of target (triangle)
        #   k_j = key of source (node)
        # You must pass edge_index as [target, source] = [tri, node]
        #
        # But your input nt_edge_index is defined as: [2, E], source=node, target=triangle
        # So: row = node (source), col = triangle (target)
        # Therefore, to get [target, source], do:
        edge_index_for_prop = torch.stack([nt_edge_index[0], nt_edge_index[1]], dim=0)

        aggr_out = self.propagate(edge_index_for_prop, q=q, k=k, v=v, size=size)
        out = self.out_proj(aggr_out) + x_tri
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

    def forward(self, x_node, x_tri, tn_edge_index, size, return_attn=False):
        """
        x_node: [N, node_dim]
        x_tri: [M, tri_dim]
        tn_edge_index: [2, E], source=triangle, target=node
        returns: [N, node_dim]
        """
        if return_attn:
            raise NotImplementedError("return_attn not supported.")

        q = self.q(x_node)   # [N, H]
        k = self.k(x_tri)    # [M, H]
        v = self.v(x_tri)    # [M, H]

        # tn_edge_index: [src=tri, dst=node]
        # For MessagePassing: need [target=node, source=tri] → already [dst, src] if we swap
        # But standard: edge_index[0] = target, edge_index[1] = source
        # Given tn_edge_index[0] = tri (source), tn_edge_index[1] = node (target)
        # So target = tn_edge_index[1], source = tn_edge_index[0]
        edge_index_for_prop = torch.stack([tn_edge_index[0], tn_edge_index[1]], dim=0)

        aggr_out = self.propagate(edge_index_for_prop, q=q, k=k, v=v, size=size)
        out = self.out_proj(aggr_out) + x_node
        return out

    def message(self, q_i, k_j, v_j, index, ptr, size_i):
        attn_logits = (q_i * k_j).sum(dim=-1) * self.scale
        alpha = softmax(attn_logits, index, num_nodes=size_i)
        return alpha.unsqueeze(-1) * v_j

    def update(self, aggr_out):
        return aggr_out


class CrossAttentionTransformer(nn.Module):
    def __init__(self, embed_dim=256, nhead=1, dropout=0.1, mlp_ratio=4, t_in=6, node=60882, triangle=115443):
        super().__init__()
        self.num_node_nodes = node
        self.num_tri_nodes = triangle
        self.t_in = t_in

        # Load static graphs (assumed same across time)
        node_edge_index, tri_edge_index, tn_edge_index, nt_edge_index = processing.generate_sparse_graph()
        self.register_buffer('node_edge_index', node_edge_index)
        self.register_buffer('tri_edge_index', tri_edge_index)
        self.register_buffer('tn_edge_index', tn_edge_index)
        self.register_buffer('nt_edge_index', nt_edge_index)

        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)

        # Attention modules (operate per time step)
        self.node_self_attn = NodeSparseSelfAttention(embed_dim, embed_dim)
        self.triangle_self_attn = TriangleSparseSelfAttention(embed_dim, embed_dim)
        self.node2tri = NodeToTriangleCrossAttention(node_dim=embed_dim, tri_dim=embed_dim, hidden_dim=embed_dim, dropout=dropout)
        self.tri2node = TriangleToNodeCrossAttention(tri_dim=embed_dim, node_dim=embed_dim, hidden_dim=embed_dim, dropout=dropout)

        # Layer norms
        self.norm_node_self = nn.LayerNorm(embed_dim)
        self.norm_triangle_self = nn.LayerNorm(embed_dim)
        self.norm_node_cross = nn.LayerNorm(embed_dim)
        self.norm_triangle_cross = nn.LayerNorm(embed_dim)
        self.norm_node_mlp = nn.LayerNorm(embed_dim)
        self.norm_triangle_mlp = nn.LayerNorm(embed_dim)

        # MLPs
        self.mlp_node = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.mlp_triangle = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, node, triangle):
        """
        node: [t, N, C]
        triangle: [t, M, C]
        returns: [t, N, C], [t, M, C]
        """
        t = node.size(0)
        node_out = []
        triangle_out = []

        for i in range(t):
            n = node[i]          # [N, C]
            tri = triangle[i]    # [M, C]

            # --- Node stream ---
            n_res = n
            n_norm = self.norm_node_self(n_res)
            n_self = self.node_self_attn(n_norm,
                                         self.node_edge_index,
                                         size=(self.num_node_nodes, self.num_node_nodes))
            n = n_res + self.dropout(n_self)

            # --- Triangle stream ---
            tri_res = tri
            tri_norm = self.norm_triangle_self(tri_res)
            tri_self = self.triangle_self_attn(tri_norm,
                                               self.tri_edge_index,
                                               size=(self.num_tri_nodes, self.num_tri_nodes))
            tri = tri_res + self.dropout(tri_self)

            # --- Cross attention: triangle <- node ---
            tri_cross = self.node2tri(
                x_tri=self.norm_triangle_cross(tri),
                x_node=self.norm_node_self(n_res),
                nt_edge_index=self.nt_edge_index,
                size=(self.num_node_nodes, self.num_tri_nodes)
            )
            tri = tri + self.dropout(tri_cross)

            # --- Cross attention: node <- triangle ---
            n_cross = self.tri2node(
                x_node=self.norm_node_cross(n),
                x_tri=self.norm_triangle_self(tri_res),
                tn_edge_index=self.tn_edge_index,
                size=(self.num_tri_nodes, self.num_node_nodes)
            )
            n = n + self.dropout(n_cross)

            # --- MLPs ---
            n = n + self.dropout(self.mlp_node(self.norm_node_mlp(n)))
            tri = tri + self.dropout(self.mlp_triangle(self.norm_triangle_mlp(tri)))

            node_out.append(n)
            triangle_out.append(tri)

        node_final = torch.stack(node_out, dim=0)        # [t, N, C]
        triangle_final = torch.stack(triangle_out, dim=0)  # [t, M, C]

        return node_final, triangle_final

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
    def __init__(self, node_var=11, triangle_var=18, embed_dim=256, t_in=3, t_out=3, dropout=0.1):
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
                 t_in=3, t_out=3,
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
    