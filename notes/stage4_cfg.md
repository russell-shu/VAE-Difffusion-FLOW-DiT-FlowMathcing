# Stage 4 · Classifier-Free Guidance (CFG)

## The trick

During training we occasionally **drop** the conditioning `y` (replace
by a special *null* token).  The same network then learns to parameterise
*both* `v_cond(z, t, y)` and `v_uncond(z, t)`.  At inference we combine
them:

```
v_CFG = v_cond + alpha * (v_cond - v_uncond)          (paper eq. 8)
```

`alpha = 0` -> unconditional, `alpha = 1` -> pure conditional, `alpha >
1` -> push harder toward class-conditional directions (the so-called
"guidance").

## Why it works (intuition)

Bayes' rule gives `p(z | y) ∝ p(z) p(y | z)`.  Differentiating the log
with respect to `z` and taking the score,

```
score(z | y) = score(z) + score(y | z)
             ≈ v_cond + (v_cond - v_uncond)
```

so guidance is scaling the "y-directional" score by `alpha`.  Large
`alpha` amplifies class-specific features -- but also over-exaggerates,
producing artifacts.  This is the *oversaturation* the paper talks about
in section 4.4; we will fix it with APG in stage 11.

## Implementation notes

* The UNet has `num_classes + 1` embeddings; the extra one is the *null
  class* used for the unconditional branch.
* `class_drop_prob = 0.1` is both the paper's training drop rate and the
  original CFG paper's recommendation.
* For the unconditional branch at sample time pass ``y=None`` -- the
  model replaces it with the null token.
* We intentionally evaluate `alpha = 0, 1, 4, 10` in the same grid so
  you can *see* the quality / saturation trade-off in one image.

## Observations to record

1. `alpha = 0` -> samples look class-agnostic (gibberish).
2. `alpha = 1` -> clean but sometimes blurry; multi-modal per class.
3. `alpha = 4` -> sharper, closer to canonical digits.
4. `alpha = 10` -> over-sharpened, bleached, cartoonish.  These are the
   artifacts the paper calls "oversaturation" (Kynkäänniemi 2024) and
   motivate APG.

## Run

```bash
python scripts/stage4_cfg_mnist.py --config configs/stage4_cfg_mnist.yaml
```

## Paper anchor

* Ho & Salimans 2021 (CFG).
* LongCat-AudioDiT eq. 8 and section 4.4 opening paragraph.
