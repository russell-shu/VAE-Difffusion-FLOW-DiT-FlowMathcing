# mini-LongCat-AudioDiT

A pedagogical re-implementation of
[LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT),
designed to make every concept in the paper (VAE, DDPM, Rectified Flow,
CFM, DiT, Euler solver, CFG, APG, REPA, masked conditioning, training-
inference mismatch fix) fall out of code.

The repo follows the 12-stage plan in
`.cursor/plans/longcat-audiodit_学习与复现规划_*.plan.md`.

```
Track A (theory, MNIST, stages 1-4):
  VAE -> DDPM -> Rectified Flow + Euler -> CFG

Track B (audio, LJSpeech/LibriTTS-R, stages 5-12):
  Audio I/O -> mini-AE -> Wav-VAE
    -> DiT + CFM (uncond)
    -> UMT5 Cross-Attn
    -> Masked conditioning
    -> Mismatch fix -> APG -> REPA -> full pipeline
```

## Layout

```
mini_audiodit/
  data/            Datasets (MNIST, LJSpeech, synthetic audio)
  models/          VAE, DDPM UNet, Wav-VAE, DiT, text encoder, REPA
  paths/           Rectified flow + DDPM schedule
  solvers/         Euler solvers (generic + TTS with mismatch fix)
  guidance/        CFG and APG
  masking/         VoiceBox-style span masking
  trainer/         Training loops per stage
  utils/           config, logging, checkpoint, viz
scripts/           One runnable entry point per stage
notes/             Per-stage learning notes
configs/           YAML configs per stage
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Every stage is a self-contained script.  Most will run on CPU in minutes
with the tiny configs shipped in `configs/`:

```bash
python scripts/stage0_smoke.py
python scripts/stage1_vae_mnist.py        --config configs/stage1_vae_mnist.yaml
python scripts/stage2_ddpm_mnist.py       --config configs/stage2_ddpm_mnist.yaml
python scripts/stage3_rf_mnist.py         --config configs/stage3_rf_mnist.yaml
python scripts/stage4_cfg_mnist.py        --config configs/stage4_cfg_mnist.yaml
python scripts/stage5_mini_ae.py          --config configs/stage5_mini_ae.yaml
python scripts/stage6_wav_vae.py          --config configs/stage6_wav_vae.yaml
python scripts/stage7_dit_uncond.py       --config configs/stage7_dit_uncond.yaml
python scripts/stage8_dit_text.py         --config configs/stage8_dit_text.yaml
python scripts/stage9_masked_tts.py       --config configs/stage9_masked_tts.yaml
python scripts/stage10_mismatch_fix.py    --config configs/stage10_mismatch_fix.yaml
python scripts/stage11_apg.py             --config configs/stage11_apg.yaml
python scripts/stage12_full_pipeline.py   --config configs/stage12_full_pipeline.yaml
```

Every stage writes artifacts (checkpoints, sample grids, audio, logs)
into `runs/<stage>/`.

Read `notes/stageN_*.md` before running stage N.  The notes explain the
theory and tell you what to observe.

## Data

- **MNIST** is fetched automatically by torchvision.
- **LJSpeech** (~2.6 GB) can be downloaded with
  `python -m mini_audiodit.data.download_ljspeech`.
- If no real audio is available, every audio stage falls back to a
  synthetic dataset (sine + FM + noise) that is large enough to trigger
  all code paths.  This is for smoke-testing only – do NOT expect the
  models to learn real speech from it.

## Paper anchors

Every stage's notes explicitly cite the equation / section / reference
in the LongCat-AudioDiT paper (`LongCat-AudioDiT.pdf` at repo root).
