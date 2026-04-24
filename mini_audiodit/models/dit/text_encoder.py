"""Multilingual text encoder -- paper section 4.2, eq. 5.

    q = LayerNorm(last_hidden_state) + LayerNorm(raw_word_embedding)

followed by a ConvNeXt V2 refinement module (F5-TTS style) that speeds
up alignment convergence.

The default backbone is ``google/umt5-base`` (paper footnote 1).  A
``use_pretrained=False`` mode falls back to a small random embedding
table so tests can run without network access / HF downloads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# ConvNeXt V2 refinement (Woo 2023).  We use 1D over the token axis.
# ---------------------------------------------------------------------------


class GRN(nn.Module):
    """Global response normalisation (ConvNeXt V2)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = x.norm(p=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block1D(nn.Module):
    def __init__(self, dim: int, kernel: int = 7, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim * mlp_ratio)
        self.act = nn.GELU()
        self.grn = GRN(dim * mlp_ratio)
        self.pwconv2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` has shape ``[B, L, D]``."""
        r = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return r + x


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------


@dataclass
class TextEncoderOutput:
    features: torch.Tensor  # [B, S, D]
    mask: torch.Tensor      # [B, S]  (1 = real token)


class TextEncoder(nn.Module):
    """Wrap UMT5 plus eq. 5 + ConvNeXt refinement."""

    def __init__(
        self,
        model_name: str = "google/umt5-base",
        refine_blocks: int = 2,
        refine_dim: int | None = None,
        use_pretrained: bool = True,
        freeze_backbone: bool = True,
        max_tokens: int = 128,
        vocab_size_fallback: int = 32_128,
        fallback_dim: int = 256,
    ) -> None:
        super().__init__()
        self.use_pretrained = use_pretrained
        self.max_tokens = max_tokens

        if use_pretrained:
            try:
                from transformers import AutoTokenizer, T5EncoderModel
            except ImportError as e:
                raise RuntimeError(
                    "transformers not installed; set use_pretrained=False for tests"
                ) from e
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.backbone = T5EncoderModel.from_pretrained(model_name)
            hidden = self.backbone.config.d_model
            if freeze_backbone:
                for p in self.backbone.parameters():
                    p.requires_grad_(False)
                self.backbone.eval()
        else:
            self.tokenizer = None
            self.backbone = nn.Embedding(vocab_size_fallback, fallback_dim)
            hidden = fallback_dim

        self.hidden_dim = hidden
        self.ln_last = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ln_raw = nn.LayerNorm(hidden, elementwise_affine=False)

        out_dim = refine_dim or hidden
        if out_dim != hidden:
            self.in_proj = nn.Linear(hidden, out_dim)
        else:
            self.in_proj = nn.Identity()
        self.refine = nn.ModuleList(
            [ConvNeXtV2Block1D(out_dim) for _ in range(refine_blocks)]
        )
        self.out_dim = out_dim

    def tokenize(self, texts: Sequence[str], device: torch.device) -> dict:
        if self.tokenizer is not None:
            toks = self.tokenizer(
                list(texts),
                padding="max_length",
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            return {k: v.to(device) for k, v in toks.items()}
        return self._fake_tokens(texts, device)

    def _fake_tokens(self, texts: Sequence[str], device: torch.device) -> dict:
        ids = []
        masks = []
        for t in texts:
            byts = t.encode("utf-8")[: self.max_tokens]
            idx = [b + 3 for b in byts]
            m = [1] * len(idx) + [0] * (self.max_tokens - len(idx))
            idx = idx + [0] * (self.max_tokens - len(idx))
            ids.append(idx)
            masks.append(m)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
        }

    def _last_and_raw(self, tokens: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = tokens["input_ids"]
        mask = tokens["attention_mask"].float()
        if self.use_pretrained:
            was_training = self.backbone.training
            if was_training:
                self.backbone.eval()
            with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
                out = self.backbone(
                    input_ids=ids,
                    attention_mask=tokens["attention_mask"],
                )
            last = out.last_hidden_state
            emb_layer = self.backbone.shared if hasattr(self.backbone, "shared") else self.backbone.get_input_embeddings()
            raw = emb_layer(ids)
            if was_training:
                self.backbone.train()
        else:
            raw = self.backbone(ids)
            last = raw
        return last, raw, mask

    def forward(self, texts: Sequence[str], device: torch.device | None = None) -> TextEncoderOutput:
        device = device or next(self.parameters()).device
        tokens = self.tokenize(texts, device)
        last, raw, mask = self._last_and_raw(tokens)
        q = self.ln_last(last) + self.ln_raw(raw)
        q = self.in_proj(q)
        for blk in self.refine:
            q = blk(q)
        return TextEncoderOutput(features=q, mask=mask)
