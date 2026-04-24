# Stage 5 · Mini 1D Audio Autoencoder

This is the *stable* baseline before we turn on the KL term, the mel
loss, and the adversarial discriminator in stage 6.

## Architecture (paper section 3.1)

| Component | What it does |
| --- | --- |
| Weight-norm 1D conv stem | projects raw waveform into a wide channel space |
| ``EncoderBlock`` stack | dilated residual units + strided conv downsample |
| Non-parametric shortcut | space-to-channel reshape + channel averaging |
| ``DecoderBlock`` stack | transpose-conv upsample + dilated residual units |
| Snake activation | periodic non-linearity tuned per channel |

The shortcut is the trick from Wu et al. 2025 (cited in the paper) --
it gives the encoder a **linear** residual path so aggressive
downsampling does not blow up training.

## Objective (subset of paper eq. 2)

```
L = lambda_time * L1(x, x_hat) + lambda_stft * L_STFT(x, x_hat)
```

No KL, no GAN yet.

## Latent frame rate

The shipped config uses strides ``[4, 8, 8, 8]`` so the cumulative
downsample ratio is ``4 * 8 * 8 * 8 = 2048``.  At 24 kHz this yields
``24000 / 2048 ≈ 11.72`` latent frames per second -- the default in
the LongCat-AudioDiT paper (section 5.1).

## What to listen for

After ~2 k steps on synthetic audio you should hear a clean
reconstruction (it will still sound synthetic because the dataset is
synthetic).  On real LJSpeech you should hear intelligible speech with
mild muffling before stage 6.

## Run

```bash
python scripts/stage5_mini_ae.py --config configs/stage5_mini_ae.yaml
```

Artifacts:

* ``gt_*.wav`` / ``rec_*.wav`` -- ground truth vs reconstruction
* ``rec_mel_*.png`` -- mel of the reconstruction

## Paper anchor

LongCat-AudioDiT section 3.1 (architecture) and the subset of eq. 2
that excludes ``L_KL``, ``L_adv``, ``L_fm``.
