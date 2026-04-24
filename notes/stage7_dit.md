# Stage 7 · DiT + CFM on Wav-VAE latents (unconditional)

First time the paper's main network appears in the code.

## Architecture summary (paper section 4.1, figure 2 left)

```
Input  [B, C=latent_dim, T=frames]
  |
  Conv1d patch embedding (patch_size=1 here)   -> [B, L, D]
  |
  for each DiT block:
      x = x + gate1 * SelfAttn( AdaLN(x, shift1, scale1), RoPE, QK-Norm )
      x = x + CrossAttn( LayerNorm(x), text )          # stage 8 only
      x = x + gate2 * MLP( AdaLN(x, shift2, scale2) )
  |
  + long-skip from patch-embedded input
  |
  LayerNorm(elementwise=False) * (1+scale) + shift     (final AdaLN)
  Linear(D -> C) and unpatch  -> [B, C, T]
```

Ingredients wired in by stage 7:

* **AdaLN-Zero** (`DiTBlock.forward`): the network starts as identity
  because the AdaLN projection is zero-initialised.  Training then
  wakes up the residual branches.
* **Global AdaLN** (`use_global_adaln=True`): one shared projection
  `SiLU -> Linear(D -> 6D)` is used by *every* block.  Paper section
  4.1: "replaces individual AdaLN projections with a shared, global
  block for all DiT layers ... significantly reduces the overall
  parameter count."
* **Long-skip connection** (`use_long_skip=True`): input tokens are
  added to the final hidden state -- a DiTTo-TTS trick cited in the
  same paragraph.
* **RoPE** (`build_rope` / `apply_rope`): relative positional encoding
  on Q and K only (no learnable positional embeddings).
* **QK-Norm**: `RMSNorm` on Q and K *after* the RoPE rotation is fine
  for sequence length stability (Henry 2020).

## How we train here

This stage is **unconditional** -- no text, no prompt, no mask.  For
every batch:

1. `wave -> Wav-VAE encoder -> z1`  (frozen).
2. `z0 ~ N(0, I)`, `t ~ U(1e-5, 1-1e-5)`.
3. `z_t = (1-t) z0 + t z1`.
4. Predict velocity `v = DiT(z_t, t)`.
5. `loss = || v - (z1 - z0) ||^2`.

Sampling uses the generic Euler solver from stage 3.  The decoded wave
will not be speech-like even on real LJSpeech at this scale; the point
is to verify that the whole `wave -> latent -> DiT -> latent -> wave`
path is wired up correctly and the loss monotonically decreases.

## Parameter counts

With the default config (`dim=128`, `num_blocks=4`, `num_heads=4`)
the DiT is ~1 M parameters.  Scaling up:

| Config | Params |
| --- | --- |
| 128 x 4  | ~1 M   |
| 384 x 12 | ~25 M  |
| 640 x 20 | ~95 M  |

The paper's 3.5 B model is `dim ~= 2048`, many blocks; qualitatively
the code is the same, just bigger.

## Validation checks

1. Training loss should drop and level out.  At 2 k steps on synthetic
   data with `dim=128 / 4 blocks`, it should go from ~1.0 to ~0.4.
2. `sample_mel_*.png` should show *some* non-trivial structure
   (banding / harmonics) rather than flat noise.  Full speech-like
   structure is out of reach at this scale; check stage 12.

## Run

Make sure stage 6 produced ``runs/stage6/model.pt`` (or allow the
warning -- the code will run with a random Wav-VAE so you can verify
the plumbing):

```bash
python scripts/stage7_dit_uncond.py --config configs/stage7_dit_uncond.yaml
```

## Paper anchor

* Paper section 4.1, figure 2 left.
* Peebles & Xie 2023 (DiT).
* Perez 2018 (AdaLN), Su 2024 (RoPE), Henry 2020 (QK-norm),
  Zhang & Sennrich 2019 (RMSNorm), Chen 2024b (global AdaLN),
  Lee 2024 / DiTTo-TTS (long-skip).
