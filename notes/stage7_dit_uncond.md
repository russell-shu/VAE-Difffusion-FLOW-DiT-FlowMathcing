# Stage 7 · DiT + unconditional CFM on Wav-VAE latents

We finally leave MNIST.  The training target is the **continuous latent**
``mu(x)`` produced by the frozen Wav-VAE encoder (stage 6).  The flow
matching setup is identical to stage 3, but the tensor is now
``[B, D, T_lat]`` with ``D = latent_dim`` and
``T_lat = T_wave / downsample_ratio``.

## Architecture mapping (paper figure 2, middle column)

| Paper component | This repo |
| --- | --- |
| Patch embedding of noisy latent | ``Conv1d(..., kernel=stride=patch_size)`` |
| DiT blocks | :class:`DiTBlock` |
| AdaLN conditioned on ``t`` | **Global** AdaLN: one MLP maps ``t`` -> ``6 * dim * depth`` scalars, sliced per block (Gentron / paper section 4.1) |
| QK-Norm | RMSNorm on Q/K heads (Henry 2020 + Zhang 2019) |
| RoPE | :func:`rope_cache` applied inside self-attention |
| Long skip | output ``final_linear`` is added back to **input latent ``x``** (residual around the whole DiT) |

Cross-attention is wired in the block but disabled here (``cross_dim=None``).

## Training objective

Same as stage 3:

```
z0 ~ N(0, I)
z1 ~ mu(wave)
zt = (1-t) z0 + t z1
L = || (z1 - z0) - DiT(zt, t) ||^2
```

## Sampling

``EulerSolver`` integrates from ``t=0`` (noise) to ``t=1`` (data), then
``WavVAE.decode`` turns the latent back into audio.

## Run

1. Train / download a Wav-VAE checkpoint (stage 6).
2. Point ``vae.ckpt`` in the YAML to that file.
3. Execute:

```bash
python scripts/stage7_dit_uncond.py --config configs/stage7_dit_uncond.yaml
```

If the checkpoint is missing the script logs a warning and keeps going
with a random VAE -- useful for CI, useless for audio quality.

## Paper anchor

LongCat-AudioDiT section 4.1 (CFM + DiT backbone) excluding the text
encoder and masking terms.
