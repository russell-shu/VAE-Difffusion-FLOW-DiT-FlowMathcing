"""Stage 6: full Wav-VAE objective (paper eq. 2)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_audiodit.data import build_audio_loader, get_audio_dataset
from mini_audiodit.models.wav_vae import (
    FeatureMatchingLoss,
    MultiResSTFTLoss,
    MultiScaleMelLoss,
    MultiScaleSTFTDiscriminator,
    WavVAE,
    kl_gaussian,
)
from mini_audiodit.utils import (
    load_config,
    plot_latent_histogram,
    save_checkpoint,
    save_mel_spectrogram,
    save_waveform,
    set_seed,
    setup_logger,
)


def hinge_d_loss(real_logits: list[torch.Tensor], fake_logits: list[torch.Tensor]) -> torch.Tensor:
    loss = real_logits[0].new_zeros(())
    for rl, fl in zip(real_logits, fake_logits):
        loss = loss + F.relu(1.0 - rl).mean() + F.relu(1.0 + fl).mean()
    return loss / len(real_logits)


def hinge_g_loss(fake_logits: list[torch.Tensor]) -> torch.Tensor:
    loss = fake_logits[0].new_zeros(())
    for fl in fake_logits:
        loss = loss - fl.mean()
    return loss / len(fake_logits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    run_dir = ROOT / cfg.run_dir
    logger = setup_logger("stage6-wav-vae", run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = get_audio_dataset(
        name=cfg.data.name,
        root=ROOT / cfg.data.root,
        sample_rate=cfg.data.sample_rate,
        segment_seconds=cfg.data.segment_seconds,
        max_items=cfg.data.max_items,
        synthetic_items=cfg.data.synthetic_items,
    )
    loader = build_audio_loader(ds, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    gen = WavVAE(
        base_channels=cfg.model.base_channels,
        channel_mults=tuple(cfg.model.channel_mults),
        strides=tuple(cfg.model.strides),
        latent_dim=cfg.model.latent_dim,
    ).to(device)
    disc = MultiScaleSTFTDiscriminator(
        fft_sizes=tuple(cfg.discriminator.fft_sizes),
        hops=tuple(cfg.discriminator.hops),
        wins=tuple(cfg.discriminator.wins),
    ).to(device)

    stft_loss = MultiResSTFTLoss().to(device)
    mel_loss = MultiScaleMelLoss(sample_rate=cfg.data.sample_rate).to(device)
    fm_loss = FeatureMatchingLoss()

    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg.optim.lr_g, betas=(0.8, 0.99))
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg.optim.lr_d, betas=(0.8, 0.99))

    step = 0
    pbar = tqdm(total=cfg.train.max_steps, desc="stage6")
    while step < cfg.train.max_steps:
        for batch in loader:
            batch = batch.to(device)
            use_adv = step >= cfg.train.warmup_steps

            # ----- Generator -----
            opt_g.zero_grad(set_to_none=True)
            rec, z, mu, logvar = gen(batch)
            l_time = F.l1_loss(rec, batch)
            l_stft = stft_loss(rec, batch)
            l_mel = mel_loss(rec, batch)
            l_kl = kl_gaussian(mu, logvar)

            if use_adv:
                logits_fake, feats_fake = disc(rec)
                l_adv = hinge_g_loss(logits_fake)
                logits_real, feats_real = disc(batch)
                l_fm = fm_loss(feats_fake, feats_real)
            else:
                l_adv = rec.new_zeros(())
                l_fm = rec.new_zeros(())

            loss_g = (
                cfg.loss.lambda_time * l_time
                + cfg.loss.lambda_stft * l_stft
                + cfg.loss.lambda_mel * l_mel
                + cfg.loss.lambda_kl * l_kl
                + cfg.loss.lambda_adv * l_adv
                + cfg.loss.lambda_fm * l_fm
            )
            loss_g.backward()
            if cfg.train.grad_clip_g is not None:
                torch.nn.utils.clip_grad_norm_(gen.parameters(), cfg.train.grad_clip_g)
            opt_g.step()

            # ----- Discriminator -----
            if use_adv:
                opt_d.zero_grad(set_to_none=True)
                with torch.no_grad():
                    rec_det, _, _, _ = gen(batch)
                logits_real, _ = disc(batch)
                logits_fake, _ = disc(rec_det.detach())
                loss_d = hinge_d_loss(logits_real, logits_fake)
                loss_d.backward()
                if cfg.train.grad_clip_d is not None:
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), cfg.train.grad_clip_d)
                opt_d.step()
            else:
                loss_d = batch.new_zeros(())

            step += 1
            pbar.update(1)

            if step % cfg.train.log_every == 0:
                msg = (
                    f"loss_g={loss_g.item():.4f} time={l_time.item():.4f} stft={l_stft.item():.4f} "
                    f"mel={l_mel.item():.4f} kl={l_kl.item():.6f} adv={l_adv.item():.4f} fm={l_fm.item():.4f} "
                    f"loss_d={loss_d.item():.4f} use_adv={int(use_adv)}"
                )
                logger.info(f"step={step} {msg}")

            if step % cfg.train.sample_every == 0:
                with torch.no_grad():
                    b0 = batch[0:1]
                    r0, _, mu0, _ = gen(b0)
                    save_waveform(b0[0, 0], run_dir / f"gt_{step:06d}", sample_rate=cfg.data.sample_rate)
                    save_waveform(r0[0, 0], run_dir / f"rec_{step:06d}", sample_rate=cfg.data.sample_rate)
                    save_mel_spectrogram(
                        r0[0, 0], run_dir / f"rec_mel_{step:06d}", sample_rate=cfg.data.sample_rate,
                    )
                    plot_latent_histogram(mu0, run_dir / f"mu_hist_{step:06d}")

            if step >= cfg.train.max_steps:
                break
    pbar.close()

    save_checkpoint(
        run_dir / "model.pt",
        {"gen": gen.state_dict(), "disc": disc.state_dict(), "cfg": dict(cfg)},
    )
    logger.info(f"saved {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
