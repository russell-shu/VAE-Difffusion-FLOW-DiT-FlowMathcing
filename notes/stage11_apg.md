# Stage 11 · Adaptive Projection Guidance (APG)

Section 4.4 replaces the linear CFG combination with APG (Sadat et al.
2024) to fight **oversaturation** when ``alpha`` is large.

## Velocity ↔ sample domain

Given a predicted velocity ``v_t`` at state ``z_t`` and time ``t``:

```
mu_t = z_t + (1 - t) * v_t
```

This is the implied endpoint ``z_1`` if we integrated perfectly along a
straight flow.  APG operates on the **guidance residual** in ``mu``
space:

```
dmu = mu_t^cond - mu_t^uncond
```

## Orthogonal decomposition

```
dmu_parallel   = proj(dmu onto mu_t)
dmu_perp     = dmu - dmu_parallel
```

The parallel component is hypothesised to cause saturation; APG shrinks
it with ``eta`` (default 0.5) while keeping the orthogonal component
scaled by ``alpha`` (default 4.0):

```
mu_apg = mu_t + alpha * dmu_perp + eta * dmu_parallel
v_apg  = (mu_apg - z_t) / (1 - t)
```

Implementation: :func:`mini_audiodit.guidance.apg.apg_velocity`.

## Reverse momentum

APG maintains a moving average of ``dmu`` across Euler steps:

```
dmu <- dmu + beta * dmu_prev,   beta = -0.3
```

Negative ``beta`` emphasises the *current* guidance direction instead of
accumulating stale momentum.

## How to listen

Train / sample the same checkpoint twice:

1. Stage 10 script (CFG) -- harsh highs when ``alpha`` is large.
2. Stage 11 script (APG) -- same ``alpha`` but less buzz / clipping.

On synthetic data the difference is subtle; use LJSpeech for a fair
test.

## Run

```bash
python scripts/stage11_apg.py --config configs/stage11_apg.yaml
```

## Paper anchor

LongCat-AudioDiT section 4.4, equations (9) and (10), citing Sadat 2024.
