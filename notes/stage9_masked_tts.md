# Stage 9 · Masked latent conditioning (VoiceBox-style)

This stage implements the **masked** CFM objective from LongCat-AudioDiT
eq. (4) (minus the extra bookkeeping for separate prompt / generation
latent lengths -- we keep a single tensor and mask along time).

## Construction

Given the clean latent ``z1 = mu(wave)``:

1. Sample a random contiguous **generation** mask ``m`` (values in
   ``{0,1}``, shape ``[B,1,T]``).  Ones mark tokens the model must
   inpaint; zeros mark the *prompt/context* region.
2. Build the context latent ``z_ctx = z1 * (1 - m)``.  Everything
   outside the generation span keeps the ground-truth values of ``z1``.
3. Concatenate ``[z_t, z_ctx]`` along the channel axis before feeding
   the DiT (in_channels = ``2 * latent_dim``).  This mirrors how the
   paper supplies both the noisy state and the masked conditioning
   tensor to the backbone (Fig. 2, "Masked Noisy Latent" + "Masked
   Prompt").

## Loss

```
v* = z1 - z0
L = mean_{t,x} (1 - m) * || v_theta(z_t, t, z_ctx, text) - v* ||^2
```

Only the generation region contributes gradients.  The prompt region can
drift during autoregressive sampling -- that bug is exactly what stage 10
fixes.

## Classifier-free dropout

With probability ``mask.cfg_drop_prob`` we zero out **both** the text
tensor ``q`` and the context latent ``z_ctx`` (and mark every text token
as padded).  This trains an unconditional velocity field needed for CFG
/ APG later.

## Sampling (stage 9 only)

We run plain Euler integration on the **conditional** model without CFG
and **without** the mismatch correction yet:

```
z0 ~ N(0, I)
z_{t+dt} = z_t + v_theta([z_t, z_ctx], t, text) dt
wave = Decoder(z_1_hat)
```

You will likely hear metallic garbage on the prompt portion until you
enable the overwrite from stage 10 -- that is expected.

## Run

```bash
python scripts/stage9_masked_tts.py --config configs/stage9_masked_tts.yaml
```

## Paper anchor

LongCat-AudioDiT eq. (4) and the VoiceBox reference (Le et al. 2024).
