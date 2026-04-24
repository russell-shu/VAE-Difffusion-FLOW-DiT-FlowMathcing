# Stage 6 · Full Wav-VAE (paper eq. 2)

We now turn on every term in the generator objective:

```
L_gen = lambda_spec L_spec
      + lambda_mel L_mel
      + lambda_time L_time
      + lambda_KL L_KL
      + lambda_adv L_adv
      + lambda_fm L_fm
```

| Term | Implementation |
| --- | --- |
| ``L_spec`` | :class:`MultiResSTFTLoss` (Zeghidour 2021) |
| ``L_mel``   | :class:`MultiScaleMelLoss` (Kumar 2023) |
| ``L_time``  | L1 on raw waveform |
| ``L_KL``    | closed-form KL to N(0, I) on ``mu/logvar`` |
| ``L_adv``   | hinge GAN loss on multi-scale STFT magnitudes |
| ``L_fm``    | L1 between intermediate discriminator features |

The discriminator is a stack of :class:`STFTDiscriminator` heads (HiFi-GAN
style) operating on log-magnitude spectrograms.

## Warmup (paper paragraph after eq. 2)

For the first ``warmup_steps`` we **disable** ``L_adv`` and ``L_fm``.
This mirrors the paper: let the autoencoder learn a stable
reconstruction map before adversarial gradients kick in.

## What to watch

1. **KL should stay small but non-zero.**  If it collapses to ~0 while
   recon losses are still high, increase ``lambda_kl`` or reduce decoder
   capacity.
2. **After warmup, ``loss_d`` should hover near 0-2** for hinge GANs.
   If it explodes, lower ``lr_d`` or ``lambda_adv``.
3. **`mu_hist_*.png` should look Gaussian-ish.**  This is the property
   the diffusion backbone wants.

## Run

```bash
python scripts/stage6_wav_vae.py --config configs/stage6_wav_vae.yaml
```

## Paper anchor

LongCat-AudioDiT section 3.2 (training objective) and the warmup
paragraph immediately after eq. 2.
