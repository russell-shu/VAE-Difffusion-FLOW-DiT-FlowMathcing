# Stage 1 · VAE on MNIST

## What a VAE is, in one breath

A VAE models data `x` via a latent variable `z` drawn from a prior
`p(z) = N(0, I)`.  The decoder `p_theta(x|z)` is a conditional
distribution over data.  An approximate posterior
`q_phi(z|x) = N(mu(x), sigma(x)^2 I)` makes the integral tractable, and
we maximise the ELBO:

```
log p(x) >= E_q [ log p(x|z) ] - KL( q(z|x) || p(z) )
           \--------------------/   \-------------------/
             reconstruction term       latent regulariser
```

We train by **minimising** the negative ELBO.  For Bernoulli-pixel
MNIST the reconstruction term is BCE summed over pixels.  For a
diagonal-Gaussian posterior vs standard-normal prior the KL has a
closed form:

```
KL = -0.5 * sum_i (1 + log sigma_i^2 - mu_i^2 - sigma_i^2)
```

The infamous **reparameterization trick** lets gradients flow through
the stochastic sampling step:

```
z = mu + sigma * eps,   eps ~ N(0, I)
```

This is *literally the line* that reappears as the Wav-VAE bottleneck
in section 3.1 of the LongCat-AudioDiT paper:

> The continuous latent representation is sampled using the
> reparameterization trick: `z = mu + sigma ⊙ eps`, where
> `eps ~ N(0, I)`.

## What to watch for

1. **Both loss components go down.**  If KL -> 0 while recon stays high,
   you have **posterior collapse**: the decoder ignores `z`.  Lower
   `beta` or use a sharper decoder.
2. **`mu` across a validation batch is approximately N(0,1).**  Stage 1
   writes `latent_hist_*.png`; histogram should overlap the analytical
   standard-normal curve.  This is the property a *diffusion prior*
   will later exploit -- if the latent is not Gaussian enough, the
   diffusion model has to spend capacity learning the ambient
   distribution rather than the content.
3. **Samples from N(0,I) are at least vaguely digit-shaped.**  They will
   be blurry; that is normal for a vanilla VAE.

## Run

```bash
python scripts/stage1_vae_mnist.py --config configs/stage1_vae_mnist.yaml
```

Artifacts under `runs/stage1/`:

- `train.log`
- `samples_STEP.png`      -- grid of decoded random-N(0,I) samples
- `latent_hist_STEP.png`  -- histogram of `mu(x)` across a val batch
- `model.pt`              -- final weights

## Paper anchor

* Kingma & Welling 2013 -- the VAE.
* LongCat-AudioDiT section 3.1 -- Wav-VAE uses the exact same
  reparameterization.  The only differences are 1D-conv encoders and a
  far heavier training objective (covered in stages 5 and 6).
