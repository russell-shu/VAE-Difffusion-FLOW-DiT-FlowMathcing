# Stage 10 · Training-inference mismatch fix

Section 4.3 of the paper isolates a subtle bug in VoiceBox-style masked
training:

* During training the noisy latent ``z_t`` is a **linear interpolation**
  between Gaussian noise and data on **both** prompt and generation
  regions (eq. 3).
* The CFM loss, however, only supervises velocities on the generation
  span (eq. 4).  Gradients never constrain the network's prediction on
  the prompt span.
* During Euler sampling the prompt region is still updated with the
  predicted velocity, so it **drifts away** from the GT interpolation.
  That breaks the assumption the network was trained on.

## Fix (eq. 7)

After every Euler step, overwrite the prompt coordinates with the exact
GT noisy latent that training would have used:

```
z_ctx(t + dt) <- (t + dt) * z_ctx + (1 - (t + dt)) * z0_ctx
```

where ``z_ctx`` is the clean prompt latent and ``z0_ctx`` is the initial
Gaussian draw restricted to the prompt region.  Implementation:
:class:`EulerTTSSolver` in ``mini_audiodit/solvers/euler_tts.py``.

## Unconditional branch corollary

To estimate a *true* unconditional velocity you must also drop the
noisy prompt; otherwise the prompt channel leaks acoustic information.
``EulerTTSSolver`` multiplies ``z_t`` by the generation mask before the
unconditional forward when ``drop_noisy_prompt_in_uncond`` is enabled.

## CFG sampling

Stage 10 also wires classifier-free guidance during sampling:

```
v = v_cond + alpha * (v_cond - v_uncond)
```

APG replaces this linear combination in stage 11.

## Ablations

Toggle booleans in ``configs/stage10_mismatch_fix.yaml``:

* ``solver.fix_mismatch: false`` reproduces the bug.
* ``solver.drop_noisy_prompt_in_uncond: false`` weakens CFG.

You should hear cleaner prompt regions (less buzz) with the fix on real
audio.  On synthetic data the effect is subtle but the code path is the
important part.

## Run

```bash
python scripts/stage10_mismatch_fix.py --config configs/stage10_mismatch_fix.yaml
```

## Paper anchor

LongCat-AudioDiT section 4.3, equations (6) and (7), and the CFG
corollary at the end of the section.
