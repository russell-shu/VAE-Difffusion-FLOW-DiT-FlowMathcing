"""1D Diffusion Transformer for Wav-VAE latents.

Implements the LongCat-AudioDiT backbone *minus* the text encoder:

* patchify 1D latent with a strided Conv1d
* global AdaLN (Gentron / paper section 4.1)
* self-attention with QK RMSNorm (Henry 2020 + Zhang 2019)
* rotary positional embeddings on Q/K (Su 2024)
* optional cross-attention (stages 8+)
* long skip from the patch-embedded input to the final layer output
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([args.sin(), args.cos()], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class RMSNorm(nn.Module):
    def __init__(self, dim: float, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)[None, None, :, :]
    sin = emb.sin().to(dtype)[None, None, :, :]
    return cos, sin


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, qk_norm: bool = True) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, C)
        return self.proj(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, ctx_dim: int, heads: int, qk_norm: bool = True) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(ctx_dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, C = x.shape
        _, Lc, _ = ctx.shape
        q = self.q(x).reshape(B, L, self.heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(ctx).reshape(B, Lc, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q_norm(q)
        k = self.k_norm(k)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, C)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, cross_dim: int | None = None) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.cross = CrossAttention(dim, cross_dim, heads) if cross_dim is not None else None
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) if cross_dim else None

    def forward(
        self,
        x: torch.Tensor,
        shift_msa: torch.Tensor,
        scale_msa: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        cross_ctx: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa = shift_msa.unsqueeze(1)
        scale_msa = scale_msa.unsqueeze(1)
        gate_msa = gate_msa.unsqueeze(1)
        shift_mlp = shift_mlp.unsqueeze(1)
        scale_mlp = scale_mlp.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)

        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h, cos_sin=cos_sin)
        x = x + gate_msa * h

        if self.cross is not None and cross_ctx is not None:
            assert self.norm_cross is not None
            h = self.norm_cross(x)
            h = self.cross(h, cross_ctx, mask=cross_mask)
            x = x + h

        h = self.norm2(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        x = x + gate_mlp * h
        return x


class DiT1D(nn.Module):
    """Patchifies ``[B, C, T]`` latents, runs DiT blocks, unpatchifies."""

    def __init__(
        self,
        in_channels: int,
        patch_size: int = 4,
        dim: int = 512,
        depth: int = 8,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        cross_dim: int | None = None,
        time_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.dim = dim
        self.patch_embed = nn.Conv1d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 6 * depth),
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, heads, mlp_ratio=mlp_ratio, cross_dim=cross_dim) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(dim, patch_size * in_channels)
        self.time_dim = time_dim
        self.depth = depth

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_ctx: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
        repa_layer: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, C, T = x.shape
        assert C == self.in_channels
        if T % self.patch_size != 0:
            pad = self.patch_size - (T % self.patch_size)
            x = F.pad(x, (0, pad))

        h0 = self.patch_embed(x).transpose(1, 2)
        tokens = h0.shape[1]

        time_emb = sinusoidal_time_embedding(t, self.time_dim)
        ada = self.time_mlp(time_emb).view(B, self.depth, 6, self.dim)

        cos_sin = rope_cache(tokens, self.dim // self.blocks[0].attn.heads, x.device, x.dtype)
        h = h0
        repa_feat: torch.Tensor | None = None
        for i, blk in enumerate(self.blocks):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada[:, i].unbind(dim=1)
            ctx = cross_ctx if blk.cross is not None else None
            cmask = cross_mask if ctx is not None else None
            h = blk(
                h,
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                cos_sin=cos_sin,
                cross_ctx=ctx,
                cross_mask=cmask,
            )
            if repa_layer is not None and i == repa_layer:
                repa_feat = h

        h = self.final_norm(h)
        h = self.final_linear(h)
        h = h.transpose(1, 2).contiguous().view(B, C, -1)
        h = h[..., :T]
        out = h + x
        if repa_layer is None:
            return out
        assert repa_feat is not None
        return out, repa_feat
