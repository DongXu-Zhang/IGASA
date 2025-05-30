import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

#from models.transformer.output_layer import AttentionOutput


class LinearMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, normalize=True):
        super(LinearMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.normalize = normalize
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

    def forward(self, input_q, input_k, input_v):
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)
        
        q, k = F.elu(q,1.) + 1., F.elu(k,1.) + 1.
        hidden_states = torch.matmul(q, torch.einsum('bhmc,bhmd->bhcd', k, v))
        if self.normalize:
            hidden_states = hidden_states / (torch.matmul(q, k.sum(2).unsqueeze(-1)) + 1e-4)
        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')
        return hidden_states

ACT_LAYERS = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.1),
    'sigmoid': nn.Sigmoid(),
    'softplus': nn.Softplus(),
    'tanh': nn.Tanh(),
    'elu': nn.ELU(),
    'gelu': nn.GELU(),
    None: nn.Identity(),
}

# class SAIGA(nn.Module):
#     def __init__(self, d_model, dropout=None, activation_fn='relu'):
#         super(SAIGA, self).__init__()
#         self.expand = nn.Linear(d_model, d_model * 2)
#         self.activation = ACT_LAYERS[activation_fn]
#         self.squeeze = nn.Linear(d_model * 2, d_model)
#         if dropout is None or dropout <= 0:
#             self.dropout =  nn.Identity()
#         else:
#             self.dropout =  nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, input_states):
#         hidden_states = self.expand(input_states)
#         hidden_states = self.activation(hidden_states)
#         hidden_states = self.squeeze(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         output_states = self.norm(input_states + hidden_states)
#         return output_states

# class SGIRA(nn.Module):
#     def __init__(self, d_model, num_heads, dropout=None, normalize=True):
#         super(SGIRA, self).__init__()
#         self.attention = LinearMultiHeadAttention(d_model, num_heads, normalize)
#         self.linear = nn.Linear(d_model, d_model)
#         if dropout is None or dropout <= 0:
#             self.dropout = nn.Identity()
#         else: self.dropout = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, input_states, memory_states):
#         hidden_states = self.attention(input_states, memory_states, memory_states)
#         hidden_states = self.linear(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         output_states = self.norm(hidden_states + input_states)
#         return output_states

class SAIGA(nn.Module):
    """
    Skip-Augmented Intrinsic Geometric Attention (SAIGA).

    This module removes dropout functionality, simplifying the design while keeping core features intact.

    Args:
        d_model (int): Dimension of input features.
        num_heads (int): Number of attention heads for geometric attention.
        activation_fn (str): Activation function name from ACT_LAYERS.
        alpha (float): Scaling factor for geometric compensation.
    """
    def __init__(self, d_model, num_heads, activation_fn='relu', alpha=0.1):
        super(SAIGA, self).__init__()

        # Main components
        self.expand = nn.Linear(d_model, d_model * 2)  # Expand dimension
        self.activation = ACT_LAYERS[activation_fn]  # Apply activation function
        self.squeeze = nn.Linear(d_model * 2, d_model)  # Reduce dimension back to original
        self.norm = nn.LayerNorm(d_model)  # Norm layer for residual connection

        # Geometric attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Learnable parameter for geometric distance compensation
        self.alpha = nn.Parameter(torch.tensor(alpha))  # Learnable scaling factor
        self.num_heads = 4  #num_heads
        self.d_model = d_model

    def forward(self, input_states, positions=None):
        """
        Forward pass of the SAIGA module.

        Args:
            input_states (torch.Tensor): Input features of shape (batch_size, seq_len, d_model).
            positions (torch.Tensor, optional): 3D positions of shape (batch_size, seq_len, 3).

        Returns:
            torch.Tensor: Output features of the same shape as input_states (batch_size, seq_len, d_model).
        """
        # 1. Expand -> Activate -> Squeeze
        hidden_states = self.expand(input_states)
        hidden_states = self.activation(hidden_states)
        squeezed_states = self.squeeze(hidden_states)

        # 2. Residual connection with normalization
        se_output = self.norm(input_states + squeezed_states)

        # 3. Geometric Attention
        q_geo = self.q_proj(se_output)  # Query projection
        k_geo = self.k_proj(se_output)  # Key projection
        v_geo = self.v_proj(se_output)  # Value projection

        # Reshape for multi-head attention
        batch_size, seq_len, _ = q_geo.size()
        head_dim = self.d_model // self.num_heads

        q_geo = q_geo.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
        k_geo = k_geo.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # (B, num_heads, seq_len, head_dim)
        v_geo = v_geo.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)  # (B, num_heads, seq_len, head_dim)

        # Semantic similarity (dot product attention scores)
        geo_similarity = torch.matmul(q_geo, k_geo.transpose(-1, -2)) / (head_dim ** 0.5)

        # Geometric distance compensation
        if positions is not None:
            # Compute pairwise Euclidean distances across sequence length
            dist = torch.cdist(positions, positions, p=2) ** 2
            geo_distance = -self.alpha * dist.unsqueeze(1)  # Scale distances with alpha
        else:
            geo_distance = 0  # Default if no positions provided

        # Combine attention scores
        geo_scores = geo_similarity + geo_distance  # Aggregate
        geo_weights = torch.softmax(geo_scores, dim=-1)  # Normalize attention weights

        # Aggregate values using attention weights
        context = torch.matmul(geo_weights, v_geo)  # Weighted sum
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # Reshape

        # 4. Residual connection and normalization
        output_states = self.norm(se_output + context)

        return output_states



class SGIRA(nn.Module):
    """
    SGIRA: Sophisticated Gated Interactive Residual Attention Module
    一个高级的 Transformer 风格模块，融合多头自注意力、跨注意力、前馈网络和残差连接。
    """
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, normalize=True, activation='gelu'):
        """
        初始化 SGIRA 模块。

        参数:
            d_model (int): 输入和输出的特征维度。
            num_heads (int): 多头注意力中的头数。
            d_ff (int, optional): 前馈网络的隐藏层维度，默认设为 4 * d_model。
            dropout (float): Dropout 概率，默认 0.1。
            normalize (bool): 是否在注意力机制中归一化，默认 True。
            activation (str): 前馈网络的激活函数，可选 'gelu' 或 'relu'，默认 'gelu'。
        """
        super(SGIRA, self).__init__()

        # 多头自注意力层（Self-Attention）
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # 多头跨注意力层（Cross-Attention），用于处理 input_states 和 memory_states
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # 前馈网络（Feed-Forward Network）
        d_ff = d_ff or 4 * d_model  # 如果未指定 d_ff，默认设为 4 倍 d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),  # 第一层线性变换，扩展维度
            nn.GELU() if activation == 'gelu' else nn.ReLU(),  # 激活函数
            nn.Dropout(dropout),       # Dropout 层
            nn.Linear(d_ff, d_model)   # 第二层线性变换，恢复维度
        )

        # Dropout 层
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 层归一化（LayerNorm），分别用于自注意力、跨注意力和前馈网络的输出
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        # 可学习的门控参数，用于融合自注意力和跨注意力
        self.gate = nn.Parameter(torch.ones(1))  # 初始化为 1，标量参数

        # 保存配置
        self.d_model = d_model
        self.num_heads = num_heads
        self.normalize = normalize

    def forward(self, input_states, memory_states, self_attn_mask=None, cross_attn_mask=None):
        """
        前向传播函数。

        参数:
            input_states (Tensor): 输入状态张量，形状 (batch_size, seq_len, d_model)。
            memory_states (Tensor): 记忆状态张量，形状 (batch_size, mem_len, d_model)。
            self_attn_mask (Tensor, optional): 自注意力掩码，形状 (seq_len, seq_len)。
            cross_attn_mask (Tensor, optional): 跨注意力掩码，形状 (seq_len, mem_len)。

        返回:
            output_states (Tensor): 输出状态张量，形状 (batch_size, seq_len, d_model)。
        """
        # 1. 自注意力（Self-Attention）
        # 计算输入序列内部的注意力关系
        self_attn_output, _ = self.self_attention(
            query=input_states,
            key=input_states,
            value=input_states,
            attn_mask=self_attn_mask
        )
        self_attn_output = self.dropout(self_attn_output)
        # 残差连接 + 层归一化
        self_states = self.norm_self(input_states + self_attn_output)

        # 2. 跨注意力（Cross-Attention）
        # 使用 input_states 作为查询，memory_states 作为键和值
        cross_attn_output, _ = self.cross_attention(
            query=self_states,
            key=memory_states,
            value=memory_states,
            attn_mask=cross_attn_mask
        )
        cross_attn_output = self.dropout(cross_attn_output)
        # 残差连接 + 层归一化
        cross_states = self.norm_cross(self_states + cross_attn_output)

        # 3. 门控融合（Gated Fusion）
        # 使用可学习的 gate 参数融合自注意力和跨注意力的结果
        fused_states = self.gate * self_states + (1 - self.gate) * cross_states

        # 4. 前馈网络（Feed-Forward Network）
        ffn_output = self.ffn(fused_states)
        ffn_output = self.dropout(ffn_output)
        # 残差连接 + 层归一化
        output_states = self.norm_ffn(fused_states + ffn_output)

        return output_states



class LinearTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout, activation_fn='relu', normalize=True):
        super(LinearTransformerLayer, self).__init__()
        self.sgira = SGIRA(d_model, num_heads, dropout, normalize)
        self.saiga = SAIGA(d_model, dropout, activation_fn)

    def forward(self, input_states, memory_states):
        hidden_states = self.sgira(input_states, memory_states)
        output_states = self.saiga(hidden_states)
        return output_states
