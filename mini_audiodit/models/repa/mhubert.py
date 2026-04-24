"""Frozen mHuBERT / HuBERT feature extractor for REPA."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class MHuBERTExtractor(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/hubert-large-ls960-ft",
        target_sr: int = 16000,
        layer: int = 8,
    ) -> None:
        super().__init__()
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.target_sr = target_sr
        self.layer = layer
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """``wave`` is ``[B, 1, T]`` at ``sample_rate`` Hz.

        Returns ``[B, T', H]`` HuBERT hidden states from ``layer``.
        """
        if wave.dim() == 2:
            wave = wave.unsqueeze(1)
        wave = wave[:, 0]
        if sample_rate != self.target_sr:
            wave = torchaudio.functional.resample(wave, sample_rate, self.target_sr)

        feats = self.processor(wave.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt")
        input_values = feats["input_values"].to(wave.device)
        attention_mask = feats.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(wave.device)

        out = self.model(input_values, attention_mask=attention_mask, output_hidden_states=True)
        hs_list = out.hidden_states
        idx = min(self.layer, len(hs_list) - 1)
        return hs_list[idx]
