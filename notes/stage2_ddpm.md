# Stage 2 · DDPM on MNIST  (the pedagogical foil)

LongCat-AudioDiT **does not use** DDPM.  We implement it here only so
that when we get to CFM / rectified flow in stage 3 we can *feel* why
the paper preferred them.

## Forward process

Given a clean sample `x_0`, choose a variance schedule
`beta_1, ..., beta_T` and define

```
q(x_t | x_{t-1}) = N(sqrt(1 - beta_t) x_{t-1}, beta_t I)
```

Because all links are Gaussian there is a closed form:

```
q(x_t | x_0) = N(sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I)
```

with `alpha_t = 1 - beta_t` and `alpha_bar_t = prod_{s<=t} alpha_s`.
This is `q_sample` in `paths/ddpm_schedule.py`.

## Training

We sample `t ~ Uniform{1..T}`, `eps ~ N(0, I)`, form

```
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps
```

and train a network `eps_theta(x_t, t)` to regress the noise:

```
L = E [ || eps - eps_theta(x_t, t) ||^2 ]
```

## Sampling (reverse SDE / DDPM ancestral)

Starting from `x_T ~ N(0, I)` we iterate

```
mu_t = (1/sqrt(alpha_t)) * ( x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta )
x_{t-1} = mu_t + sqrt(beta_t) * z,   z ~ N(0, I)   (z = 0 at t = 1)
```

That is in `DDPMSchedule.ancestral_sample`.  Notice how much
bookkeeping we need: `betas`, `alphas`, `alpha_bar`, `sqrt(alpha_bar)`,
`sqrt(1 - alpha_bar)` -- six tensors driven by a discrete schedule.

## DDPM ↔ rectified flow correspondence

The continuous-time SDE view (Song 2020) shows that the DDPM forward
process is the same family as *stochastic interpolants* (Albergo 2025,
cited by the paper).  For a **variance-preserving** schedule:

```
x_t = alpha(t) x_0 + sigma(t) eps,   alpha^2 + sigma^2 = 1
```

For rectified flow (the LongCat-AudioDiT choice, eq. 3) the
interpolation is linear in `t` with `(alpha, sigma) = (1-t, t)` applied
to `(data, noise)`:

```
z_t = (1 - t) z_0 + t z_1,   z_0 ~ data, z_1 ~ N(0, I)
```

(The paper's `z_0` / `z_1` convention flips "data" and "noise" relative
to most diffusion references; see their eq. 3 and footnote in stage 3's
note.)

Both paths are straight or curved trajectories between `data` and
`noise`; DDPM with a specific schedule is one particular trajectory.
What changes is the **training objective**:

| Framework | Predicts | Loss |
| --- | --- | --- |
| DDPM      | noise `eps` | `|| eps - eps_theta ||^2` |
| Flow Matching | velocity `v = d x_t / d t` | `|| v - v_theta ||^2` |

Flow matching picks the most natural target for an ODE sampler, which
is why section 2.1 of the paper calls it simpler and schedule-free.

## What to watch for

1. Loss should drop from ~1.0 to a much smaller plateau.
2. `samples_*.png` should contain digit-shaped blobs after a few
   thousand steps.  With 4 k steps and 200-step sampling on CPU it is
   still blurry -- that's fine.
3. Notice that sampling *quality* depends heavily on
   `sample.sample_steps`.  Drop it to 20 and digits collapse.  In
   stage 3 we will see that flow matching tolerates far fewer steps.

## Run

```bash
python scripts/stage2_ddpm_mnist.py --config configs/stage2_ddpm_mnist.yaml
```

## Paper anchor

* Ho 2020 (DDPM), Sohl-Dickstein 2015.
* Albergo 2025 -- the unifying framework that makes DDPM and flow
  matching equivalent in expressive power.  The paper cites it in
  section 2.1 to justify preferring CFM on practical grounds.
