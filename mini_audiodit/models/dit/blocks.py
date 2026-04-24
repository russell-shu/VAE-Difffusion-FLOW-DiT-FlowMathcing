"""DiT blocks with AdaLN-Zero, RoPE, QK-Norm, and optional cross-attention.

Matches Figure 2 (left) of LongCat-AudioDiT:

    x ---> AdaLN --> Self-Attn  --> Scale --> +
      |                                        ^
      +----------------------------------------+
      |
      v
      LayerNorm --> Cross-Attn --> Scale --> +
                                              ^
      +---------------------------------------+
      v
      AdaLN --> MLP --> Scale --> +
                                   ^
      +-----------------------------+
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """LongCat-AudioDiT uses RMSNorm for QK-norm (paper sect. 4.1)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight


def build_rope(head_dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute (cos, sin) tables for RoPE.

    Returns a tensor of shape ``[max_len, head_dim // 2, 2]`` with
    ``[..., 0] = cos`` and ``[..., 1] = sin``.
    """
    assert head_dim % 2 == 0, "RoPE requires head_dim divisible by 2"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    angles = t.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.stack([angles.cos(), angles.sin()], dim=-1)


def apply_rope(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to the last dim of ``x``.  ``x`` is ``[..., L, Dh]``."""
    L = x.shape[-2]
    cos = rope[:L, :, 0]
    sin = rope[:L, :, 1]
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot = torch.stack(
        [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1
    )
    return rot.flatten(-2)


class SelfAttentionRoPE(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, rope)
        k = apply_rope(k, rope)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, L, D))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, kv_dim: int | None = None) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        kv_dim = kv_dim or dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(kv_dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, L, D = x.shape
        S = context.shape[1]
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).view(B, S, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if context_mask is not None:
            attn_mask = context_mask[:, None, None, :].to(dtype=q.dtype)
            attn_mask = (1.0 - attn_mask) * -1e4
        else:
            attn_mask = None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        cross_attn_kv_dim: int | None = None,
        use_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.self_attn = SelfAttentionRoPE(dim, num_heads)
        if use_cross_attn:
            self.norm_ca = nn.LayerNorm(dim)
            self.cross_attn = CrossAttention(dim, num_heads, kv_dim=cross_attn_kv_dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, hidden)

    def forward(
        self,
        x: torch.Tensor,
        ada: torch.Tensor,
        rope: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = ada.chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift1, scale1)
        h = self.self_attn(h, rope=rope)
        x = x + gate1.unsqueeze(1) * h
        if self.use_cross_attn and context is not None:
            x = x + self.cross_attn(self.norm_ca(x), context, context_mask)
        h = modulate(self.norm2(x), shift2, scale2)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h
        return x
