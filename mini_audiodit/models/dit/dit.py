"""DiT wrapper around 1D latent sequences -- LongCat-AudioDiT section 4.1."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .blocks import DiTBlock, build_rope
from ..unet_2d import sinusoidal_embedding


class DiT(nn.Module):
    """Diffusion Transformer for 1D latent sequences.

    Options matching the paper's configuration:

    * ``use_global_adaln=True``  -- shared AdaLN projection for all blocks
      (Gentron; paper section 4.1 "global AdaLN").
    * ``use_long_skip=True``     -- input added to final hidden state.
    * ``use_cross_attn``         -- enable cross-attention to a text / context
      tensor of shape ``[B, S, text_dim]``.

    ``forward`` expects ``x`` of shape ``[B, C, T]`` and returns velocity
    of the same shape (CFM convention).
    """

    def __init__(
        self,
        input_dim: int,
        dim: int = 256,
        num_blocks: int = 8,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        max_seq_len: int = 4096,
        use_long_skip: bool = True,
        use_global_adaln: bool = True,
        use_cross_attn: bool = False,
        text_dim: int | None = None,
        time_emb_dim: int = 256,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.input_dim = input_dim
        self.dim = dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.patch_size = patch_size
        self.use_long_skip = use_long_skip
        self.use_global_adaln = use_global_adaln
        self.use_cross_attn = use_cross_attn
        self.time_emb_dim = time_emb_dim

        self.patch_emb = nn.Conv1d(input_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        if use_global_adaln:
            self.global_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        else:
            self.global_ada = None
            self.per_block_ada = nn.ModuleList(
                [nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim)) for _ in range(num_blocks)]
            )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    cross_attn_kv_dim=text_dim,
                    use_cross_attn=use_cross_attn,
                )
                for _ in range(num_blocks)
            ]
        )

        rope = build_rope(self.head_dim, max_seq_len)
        self.register_buffer("rope", rope, persistent=False)

        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out_proj = nn.Linear(dim, input_dim * patch_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.patch_emb.weight, std=0.02)
        nn.init.zeros_(self.patch_emb.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        if self.global_ada is not None:
            nn.init.zeros_(self.global_ada[-1].weight)
            nn.init.zeros_(self.global_ada[-1].bias)
        else:
            for m in self.per_block_ada:
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(sinusoidal_embedding(t, self.time_emb_dim))

    def _pad_to_patch(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        T = x.shape[-1]
        pad = (-T) % self.patch_size
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        return x, pad

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, C, T_in = x.shape
        x_padded, pad = self._pad_to_patch(x)
        h = self.patch_emb(x_padded).transpose(1, 2)
        L = h.shape[1]
        if L > self.rope.shape[0]:
            raise ValueError(
                f"Sequence length {L} exceeds RoPE max {self.rope.shape[0]}; "
                f"increase dit.max_seq_len."
            )
        rope = self.rope[:L]

        t_emb = self._time_embedding(t)
        skip = h if self.use_long_skip else None

        for i, block in enumerate(self.blocks):
            ada = self.global_ada(t_emb) if self.use_global_adaln else self.per_block_ada[i](t_emb)
            h = block(h, ada=ada, rope=rope, context=context, context_mask=context_mask)

        if skip is not None:
            h = h + skip

        shift, scale = self.final_ada(t_emb).chunk(2, dim=-1)
        h = self.norm_out(h) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        h = self.out_proj(h)
        y = h.reshape(B, L, self.patch_size, self.input_dim).permute(0, 3, 1, 2).reshape(B, self.input_dim, L * self.patch_size)
        if pad:
            y = y[..., :T_in]
        return y
