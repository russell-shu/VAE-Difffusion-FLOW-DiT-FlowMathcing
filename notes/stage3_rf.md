# Stage 3 · Rectified Flow + Euler ODE solver

This is the single most important stage for the paper.  Equations 3, 4,
and 6 of LongCat-AudioDiT live here verbatim.

## Setup

Under the paper convention (watch the indexing):

```
z_0 ~ N(0, I)        (noise)
z_1 ~ p_data          (clean target)
z_t = (1 - t) z_0 + t z_1,   t in [0, 1]           <-- eq. 3
```

Differentiate w.r.t. `t`:

```
d z_t / dt = z_1 - z_0
```

This is a constant-in-time velocity **along each trajectory**.  The
whole rectified-flow idea (Liu 2022a) is to **learn a neural
approximation `v_theta(z_t, t)`** of this velocity, so that at
inference we can go noise -> data by integrating an ODE that no longer
needs access to `z_1`.

## Training objective (eq. 4, unmasked form)

```
L = E_{t, z_0, z_1} || (z_1 - z_0) - v_theta(z_t, t) ||^2
```

That's it -- no noise schedule, no beta, no alpha_bar.  **Schedule-free.**

## Euler sampler (eq. 6, unmasked)

```
z_{t + dt} = z_t + v_theta(z_t, t) * dt
```

with `t = 0 ... 1` divided into NFE steps.  Stage 3 writes one grid per
NFE in `runs/stage3/samples_*_nfe<N>.png`.  You should see:

| NFE | Expected visual |
| --- | --- |
|  8  | digits recognisable but mushy |
| 16  | sharper, minor artifacts |
| 50  | essentially the same as 16 -- diminishing returns |

Compare to the DDPM grid from stage 2 with the same 16 steps.  Rectified
flow will look better at low NFE.  This is exactly the observation
that motivated NFE=16 in the LongCat-AudioDiT paper (section 4.3).

## How it maps to the paper

* Equations 3 and 4 -- direct copy, minus the mask `(1-m)`, which we
  introduce in stage 9.
* Equation 6 (Euler) -- the body of :class:`EulerSolver`.
* Section 2.1 "flow matching ... simpler mathematical formulation ...
  eliminating the need for complex noise scheduling" -- that's the
  one-line loss in `training_step`.

## Practical tips

* Clamp `t` away from exactly 0 and 1 during training (see
  `RectifiedFlow.sample_t`).  At `t=0` the model only ever sees noise
  with no signal; at `t=1` the interpolation is the data itself, and
  the target `z_1 - z_0` depends on the *particular* noise draw.  We
  use `[1e-5, 1 - 1e-5]`.
* Some papers reverse the convention (`t=0` data, `t=1` noise).  Make
  sure the Euler direction matches the training interpolation.  The
  paper's eq. 3 places data at `t=1`.
* You can make the sampler equivalent to Heun / midpoint by averaging
  two velocity queries.  For LongCat-AudioDiT they stuck with Euler.

## Run

```bash
python scripts/stage3_rf_mnist.py --config configs/stage3_rf_mnist.yaml
```

## Paper anchor

* Lipman 2022 (Flow matching), Liu 2022a (Rectified flow).
* LongCat-AudioDiT eq. 3, 4, 6 and section 2.1.
