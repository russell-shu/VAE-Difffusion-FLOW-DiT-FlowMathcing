"""Stage 11: APG replaces CFG during sampling (paper eq. 9-10)."""
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
from mini_audiodit.guidance import APGConfig, APGState, apg_velocity
from mini_audiodit.masking import apply_cfg_drop, random_span_mask
from mini_audiodit.models.dit1d import DiT1D
from mini_audiodit.models.text import UMT5TextEncoder
from mini_audiodit.models.wav_vae import WavVAE
from mini_audiodit.paths import RectifiedFlow
from mini_audiodit.solvers import EulerTTSSolver
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
    return pad[:, None, None, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    run_dir = ROOT / cfg.run_dir
    logger = setup_logger("stage11-apg", run_dir)

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
        in_channels=latent_ch * 2,
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

    solver = EulerTTSSolver(
        num_steps=cfg.solver.nfe,
        fix_mismatch=cfg.solver.fix_mismatch,
        drop_noisy_prompt_in_uncond=cfg.solver.drop_noisy_prompt_in_uncond,
    )

    apg_cfg = APGConfig(
        alpha=float(cfg.apg.alpha),
        eta=float(cfg.apg.eta),
        beta=float(cfg.apg.beta),
    )

    step = 0
    pbar = tqdm(total=cfg.train.max_steps, desc="stage11")
    while step < cfg.train.max_steps:
        for wave, texts in loader:
            wave = wave.to(device)
            with torch.no_grad():
                z1 = encode_latent(vae, wave)
            b, _, t = z1.shape
            m = random_span_mask(
                t, b, device,
                min_gen_frac=cfg.mask.min_gen_frac,
                max_gen_frac=cfg.mask.max_gen_frac,
            )
            z_ctx = z1 * (1.0 - m)

            q, pad = text_enc(list(texts))
            q, pad, z_ctx = apply_cfg_drop(q, pad, z_ctx, cfg.mask.cfg_drop_prob)

            z0 = RectifiedFlow.draw_prior(z1.shape, device=device)
            tt = RectifiedFlow.sample_t(b, device=device)
            z_t = RectifiedFlow.interpolate(z0, z1, tt)
            v_tgt = RectifiedFlow.target_velocity(z0, z1)

            x_t = torch.cat([z_t, z_ctx], dim=1)
            cmask = cross_attn_mask(pad)

            optim.zero_grad(set_to_none=True)
            v_pred = dit(x_t, tt, cross_ctx=q, cross_mask=cmask)
            diff = (v_pred - v_tgt).pow(2) * (1.0 - m)
            denom = (1.0 - m).sum().clamp_min(1.0)
            loss = diff.sum() / denom
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
                    apg_state = APGState(apg_cfg)
                    prompt = [cfg.sample.prompt]
                    q0, pad0 = text_enc(prompt)
                    z_ctx0 = z1[:1] * (1.0 - m[:1])
                    z_noise = RectifiedFlow.draw_prior(z1[:1].shape, device=device)

                    def velocity_fn(z, t, z_ctx_in, cond):
                        ctx = z_ctx_in if z_ctx_in is not None else torch.zeros_like(z_ctx0)
                        x_in = torch.cat([z, ctx], dim=1)
                        if cond is None:
                            return dit(x_in, t, cross_ctx=None, cross_mask=None)[0]
                        return dit(
                            x_in,
                            t,
                            cross_ctx=cond["q"],
                            cross_mask=cond["cmask"],
                        )[0]

                    def guidance(vc, vu, z, tt):
                        return apg_velocity(vc, vu, z, tt, apg_state)

                    z_hat = solver.integrate(
                        velocity_fn=velocity_fn,
                        z0=z_noise,
                        z_ctx=z_ctx0,
                        mask=m[:1],
                        cond={"q": q0, "cmask": cross_attn_mask(pad0)},
                        guidance=guidance,
                    )
                    audio = vae.decode(z_hat)
                    save_waveform(
                        audio[0, 0], run_dir / f"sample_{step:06d}", sample_rate=cfg.data.sample_rate,
                    )

            if step >= cfg.train.max_steps:
                break
    pbar.close()

    save_checkpoint(
        run_dir / "dit_apg.pt",
        {"dit": dit.state_dict(), "refine": text_enc.refine.state_dict(), "cfg": dict(cfg)},
    )
    logger.info(f"saved {run_dir / 'dit_apg.pt'}")


if __name__ == "__main__":
    main()
