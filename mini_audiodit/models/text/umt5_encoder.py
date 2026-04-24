"""UMT5 text encoder used in LongCat-AudioDiT section 4.2.

Implements eq. (5):

    q = LayerNorm(last_hidden_state) + LayerNorm(raw_word_embedding)

followed by a lightweight ConvNeXt-V2 style depthwise separable stack for
local refinement (paper cites Woo 2023).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class ConvNeXtV2Refine(nn.Module):
    """Depthwise separable Conv1d stack with residual connections."""

    def __init__(self, dim: int, kernel: int = 7, depth: int = 2, drop: float = 0.1) -> None:
        super().__init__()
        pad = kernel // 2
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [
                nn.Conv1d(dim, dim, kernel_size=kernel, padding=pad, groups=dim),
                nn.Conv1d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.Dropout(drop),
            ]
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``x`` is ``[B, L, C]``."""
        h = x.transpose(1, 2)
        h = self.net(h).transpose(1, 2)
        h = self.norm(h + x)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        return h


class UMT5TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "google/umt5-base",
        freeze: bool = True,
        refine_depth: int = 2,
        refine_kernel: int = 7,
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.t5 = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.t5.config.d_model)

        if freeze:
            for p in self.t5.parameters():
                p.requires_grad_(False)

        self.refine = ConvNeXtV2Refine(self.hidden_size, kernel=refine_kernel, depth=refine_depth)

    @property
    def device(self) -> torch.device:
        return next(self.refine.parameters()).device

    def forward(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device).bool()

        with torch.set_grad_enabled(any(p.requires_grad for p in self.t5.parameters())):
            out = self.t5(input_ids=input_ids, attention_mask=attention_mask.float())
            last = out.last_hidden_state

        emb_layer = self.t5.get_input_embeddings()
        raw = emb_layer(input_ids)

        q = F.layer_norm(last, (last.shape[-1],)) + F.layer_norm(raw, (raw.shape[-1],))
        q = self.refine(q, mask=attention_mask)

        key_padding = ~attention_mask
        return q, key_padding
