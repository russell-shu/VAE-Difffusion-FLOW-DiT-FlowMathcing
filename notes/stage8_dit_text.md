# Stage 8 · UMT5 text embeddings + cross-attention

This stage wires the **text side** of LongCat-AudioDiT section 4.2 into
the DiT from stage 7.

## Text representation (eq. 5)

```
last = T5Encoder(...).last_hidden_state
raw  = embedding_layer(input_ids)
q = LayerNorm(last) + LayerNorm(raw)
q = ConvNeXtV2Refine(q)
```

Only the lightweight refine stack is trainable if ``freeze: true`` in
the YAML (the default).  Everything else is a feature extractor exactly
like the paper's setup with ``google/umt5-base``.

## Conditioning the DiT

Each DiT block now runs:

1. global-AdaLN self-attention (RoPE + QK RMSNorm)
2. cross-attention into ``q`` with a padding mask

The mask is ``True`` on padded token positions so softmax weights there
are zero.

## Training objective

Still unconditional flow matching on the **entire** latent:

```
L = || (z1 - z0) - DiT(zt, t, text=q) ||^2
```

No span masking yet -- that is stage 9.

## HuggingFace weights

The first run downloads ~1 GB of UMT5 weights.  Make sure you have disk
space and (optionally) set ``HF_HOME`` to a fast SSD.

## Run

```bash
python scripts/stage8_dit_text.py --config configs/stage8_dit_text.yaml
```

During sampling the script fixes a single English prompt from the YAML
(``sample.prompt``) so you can listen for coarse alignment even on the
synthetic audio regime.

## Paper anchor

LongCat-AudioDiT section 4.2 and eq. (5).
