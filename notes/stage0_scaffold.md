# Stage 0 · Scaffolding, Audio I/O, Visualisation

Nothing paper-specific yet; this stage just removes all the boring
boilerplate so that stages 1-12 can focus on one concept each.

## What you get

| Module | Purpose |
| --- | --- |
| `mini_audiodit/utils/config.py`      | YAML loader with attribute access |
| `mini_audiodit/utils/logging.py`     | stdout + file logger |
| `mini_audiodit/utils/seed.py`        | deterministic seeding |
| `mini_audiodit/utils/checkpoint.py`  | save / load torch state dicts |
| `mini_audiodit/utils/viz.py`         | image grids, waveforms, mel-spec, latent histograms |
| `mini_audiodit/trainer/base.py`      | generic training loop (step-based, not epoch-based) |
| `mini_audiodit/data/mnist.py`        | MNIST loaders |
| `mini_audiodit/data/audio.py`        | `LJSpeechDataset`, `TextAudioDataset`, and `SyntheticAudioDataset` fallback |

## The synthetic audio fallback

Every audio stage (5-12) accepts a dataset name:

* `dataset: synthetic` -- procedural sine + FM + noise mixtures.  Runs on
  CPU, no downloads.  Use this to check all code paths end-to-end.
* `dataset: ljspeech`  -- real speech after running
  `python -m mini_audiodit.data.download_ljspeech data/raw`.

The synthetic fallback is intentionally *not* speech.  It is enough to
show that the Wav-VAE can reconstruct its input, that the flow-matching
loss goes down, and that samplers produce something at all.  Do not
expect intelligible speech out of it.

## Validation checklist

Run the smoke test:

```bash
python scripts/stage0_smoke.py
```

It should print the shapes of one MNIST and one synthetic audio batch
and write a waveform + mel-spec PNG under `runs/stage0/`.

## Paper anchor

Section 5.1 ("Experimental Setup") lists the real dataset sizes
(200 K h for Wav-VAE, 100 K h for TTS) and the 24 kHz sample rate.  The
scaffold here mirrors the 24 kHz default but uses much shorter clips.
