"""Stage 8: DiT + CFM with UMT5 text cross-attention (no masking yet)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_audiodit.data import build_audio_loader, get_audio_dataset, wrap_with_text
from mini_audiodit.models.dit1d import DiT1D
from mini_audiodit.models.text import UMT5TextEncoder
from mini_audiodit.models.wav_vae import WavVAE
from mini_audiodit.paths import RectifiedFlow
from mini_audiodit.solvers import EulerSolver
from mini_audiodit.utils import (
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_waveform,
    set_seed,
    setup_logger,
)


@torch.no_grad()
def encode_latent(vae: WavVAE, wave: torch.Tensor) -> torch.Tensor:
    mu, _ = vae.encode(wave)
    return mu


def cross_attn_mask(pad: torch.Tensor) -> torch.Tensor:
    """``pad`` is bool [B, Lctx] True where padded."""
    return pad[:, None, None, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    run_dir = ROOT / cfg.run_dir
    logger = setup_logger("stage8-dit-text", run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.data.name.lower() == "ljspeech":
        ds = get_audio_dataset(
            name=cfg.data.name,
            root=ROOT / cfg.data.root,
            sample_rate=cfg.data.sample_rate,
            segment_seconds=cfg.data.segment_seconds,
            max_items=cfg.data.max_items,
            synthetic_items=cfg.data.synthetic_items,
            return_text=True,
        )
    else:
        base = get_audio_dataset(
            name=cfg.data.name,
            root=ROOT / cfg.data.root,
            sample_rate=cfg.data.sample_rate,
            segment_seconds=cfg.data.segment_seconds,
            max_items=cfg.data.max_items,
            synthetic_items=cfg.data.synthetic_items,
        )
        ds = wrap_with_text(base, None)

    loader = build_audio_loader(
        ds, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers, text=True,
    )

    vae = WavVAE(
        base_channels=cfg.vae.base_channels,
        channel_mults=tuple(cfg.vae.channel_mults),
        strides=tuple(cfg.vae.strides),
        latent_dim=cfg.vae.latent_dim,
    ).to(device)
    ckpt_path = ROOT / cfg.vae.ckpt
    if ckpt_path.exists():
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        vae.load_state_dict(ckpt["gen"])
        logger.info(f"loaded Wav-VAE from {ckpt_path}")
    else:
        logger.warning(f"missing VAE ckpt {ckpt_path} -- random VAE (debug only)")
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    text_enc = UMT5TextEncoder(
        model_name=cfg.text.model_name,
        freeze=cfg.text.freeze,
        refine_depth=cfg.text.refine_depth,
        refine_kernel=cfg.text.refine_kernel,
    ).to(device)

    latent_ch = cfg.vae.latent_dim
    dit = DiT1D(
        in_channels=latent_ch,
        patch_size=cfg.dit.patch_size,
        dim=cfg.dit.dim,
        depth=cfg.dit.depth,
        heads=cfg.dit.heads,
        mlp_ratio=cfg.dit.mlp_ratio,
        cross_dim=text_enc.hidden_size,
        time_dim=cfg.dit.time_dim,
    ).to(device)

    params = list(dit.parameters()) + list(text_enc.refine.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.optim.lr)

    step = 0
    pbar = tqdm(total=cfg.train.max_steps, desc="stage8")
    while step < cfg.train.max_steps:
        for wave, texts in loader:
            wave = wave.to(device)
            with torch.no_grad():
                z1 = encode_latent(vae, wave)
            z0 = RectifiedFlow.draw_prior(z1.shape, device=device)
            t = RectifiedFlow.sample_t(z1.shape[0], device=device)
            z_t = RectifiedFlow.interpolate(z0, z1, t)
            v_tgt = RectifiedFlow.target_velocity(z0, z1)

            q, pad = text_enc(list(texts))
            cmask = cross_attn_mask(pad)

            optim.zero_grad(set_to_none=True)
            v_pred = dit(z_t, t, cross_ctx=q, cross_mask=cmask)
            loss = F.mse_loss(v_pred, v_tgt)
            loss.backward()
            if cfg.train.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(dit.parameters(), cfg.train.grad_clip)
                torch.nn.utils.clip_grad_norm_(text_enc.refine.parameters(), cfg.train.grad_clip)
            optim.step()

            step += 1
            pbar.update(1)
            if step % cfg.train.log_every == 0:
                logger.info(f"step={step} loss={loss.item():.4f}")

            if step % cfg.train.sample_every == 0:
                with torch.no_grad():
                    prompt = [cfg.sample.prompt]
                    q0, pad0 = text_enc(prompt)
                    cmask0 = cross_attn_mask(pad0)
                    z_s = RectifiedFlow.draw_prior((1, z1.shape[1], z1.shape[2]), device=device)
                    solver = EulerSolver(num_steps=cfg.sample.nfe)
                    z_s = solver.integrate(
                        lambda z, tt: dit(z, tt, cross_ctx=q0, cross_mask=cmask0)[0],
                        z0=z_s,
                    )
                    audio = vae.decode(z_s)
                    save_waveform(
                        audio[0, 0], run_dir / f"sample_{step:06d}", sample_rate=cfg.data.sample_rate,
                    )

            if step >= cfg.train.max_steps:
                break
    pbar.close()

    save_checkpoint(
        run_dir / "dit_text.pt",
        {"dit": dit.state_dict(), "refine": text_enc.refine.state_dict(), "cfg": dict(cfg)},
    )
    logger.info(f"saved {run_dir / 'dit_text.pt'}")


if __name__ == "__main__":
    main()
