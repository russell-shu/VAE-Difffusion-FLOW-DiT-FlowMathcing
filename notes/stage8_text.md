# Stage 8 · Text encoder + cross-attention

## Eq. 5 of the paper

```
q = LayerNorm(last_hidden_state) + LayerNorm(raw_word_embedding)
```

* `last_hidden_state`  -- output of the UMT5 encoder stack; carries
  contextualised semantics.
* `raw_word_embedding` -- the input embedding table (``shared`` in
  HuggingFace T5); preserves phonetic surface-form cues that the deep
  stack would otherwise abstract away.

Both are LayerNorm'd with `elementwise_affine=False` (no learnable
scale/shift) and summed.  The resulting `q` has shape `[B, S, D]` where
`D` is the backbone hidden size.

Then a small **ConvNeXt V2** stack refines `q` along the token axis.
The paper cites F5-TTS (Chen et al. 2024c) as showing these blocks
accelerate alignment convergence.

## How we plug into DiT

`DiTBlock.use_cross_attn=True` enables the cross-attention branch
between self-attention and MLP (see figure 2 left).  K/V come from the
refined text features; Q comes from the latent tokens.  Cross-attention
is QK-normed, bias-free, and uses LayerNorm (not AdaLN) on the query
side -- consistent with the paper's figure.

## Pretrained vs fallback

* `text_encoder.use_pretrained=true`  -> loads `google/umt5-base` via
  `transformers`.  Needs ~1 GB of disk and network access.  The
  backbone is frozen by default so we only train the projection,
  ConvNeXt refinement, and the DiT cross-attention weights.
* `text_encoder.use_pretrained=false` -> small random embedding table.
  Enough to smoke-test all code paths (our synthetic dataset will work
  here), **but the text "semantics" are meaningless** -- we are only
  verifying shapes and the gradient path.

## Stage 9 will add masking

Stage 8 is a "predict everything from text" model with no prompt
audio.  In real use this does not produce clean speech without a
prompt, because voice identity is underdetermined.  Stage 9 replaces
this setup with the full VoiceBox-style masked conditioning.

## Run

```bash
python scripts/stage8_dit_text.py --config configs/stage8_dit_text.yaml
```

## Paper anchor

* Paper section 4.2, eq. 5.
* Chung 2023 (UMT5), Woo 2023 (ConvNeXt V2), Chen 2024c (F5-TTS's
  refinement trick).
