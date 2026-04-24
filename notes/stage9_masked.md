# Stage 9 · Masked Conditioning (eq. 4)

This is where LongCat-AudioDiT stops being "just a generic CFM model"
and becomes a zero-shot voice cloning model.

## The trick (Le et al. 2024, VoiceBox)

For each training utterance, sample a contiguous **span** and mark it
as the *prediction* region (`m = 1`).  Outside the span is the
*prompt* region (`m = 0`).  Let `z_1` be the clean latent (output of
Wav-VAE encoder).  Then:

```
z_ctx = (1 - m) * z_1        (clean prompt, zeros on predict span)
```

The DiT is fed `[z_t, z_ctx]` concatenated along the channel axis.  It
learns to predict the velocity **only on the prediction region**:

```
L_CFM = E [ || m ⊙ ( (z_1 - z_0) - v_theta(z_t, t, z_ctx, q) ) ||^2 ]   (paper eq. 4)
```

At inference we hand the model a real prompt's latent as `z_ctx`, feed
text `q`, and ask for the missing span.  Because the model has seen
lots of examples where the prompt fully specifies the speaker, at
inference it copies the voice.

## Mask ratio distribution

VoiceBox trains with `r ~ U(0.7, 1.0)` -- the predict region is at
least 70 % of the sequence.  The paper uses similar ratios; we expose
them as `mask.min_ratio` / `mask.max_ratio`.

## Why drop `z_ctx` and `q` during training

CFG at inference needs an **unconditional** prediction, which the
training has to produce.  So with probability `0.1` each we drop the
prompt (replace `z_ctx` by zeros) and the text (zero the text features
AND the attention mask so cross-attention sees nothing).  At inference
the CFG path calls the model twice -- conditional + unconditional --
and combines velocities via eq. 8.

## DiT input shape

The DiT now takes `2 * latent_dim` channels (see the
`MaskedVelocity` wrapper in `scripts/stage9_masked_tts.py`).  Only the
first `latent_dim` channels of the prediction are used; the second half
is discarded.  The paper uses the full concat and projects back to
`latent_dim`, which is functionally equivalent.

## Sample-time prompt configuration

The sampling script builds a deterministic "first X frames = prompt,
rest = generate" mask so you can *hear* whether the model copies the
prompt's timbre.

## Validation checks

1. Training loss should drop but plateau higher than stage 8 -- the
   task is harder (the prompt restricts the solution space).
2. `sample_mel_*.png` should show the *first part of the spectrogram
   is identical to the prompt's* and the rest is generated.  On
   synthetic data the "identical" part is fuzzy (the Wav-VAE only
   encodes a rough signal), but the concept is visible.

## Run

```bash
python scripts/stage9_masked_tts.py --config configs/stage9_masked_tts.yaml
```

## Paper anchor

* Paper eq. 4, eq. 5's `q` integration into cross-attention.
* Le et al. 2024 (VoiceBox).
* Paper section 4.3 last paragraph (CFG drop probabilities).
