"""Stage 5: mini 1D audio autoencoder (no KL, no GAN)."""
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
from mini_audiodit.models.wav_vae import MultiResSTFTLoss, WavAE
from mini_audiodit.utils import (
    load_config,
    save_checkpoint,
    save_mel_spectrogram,
    save_waveform,
    set_seed,
    setup_logger,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    run_dir = ROOT / cfg.run_dir
    logger = setup_logger("stage5-mini-ae", run_dir)

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

    model = WavAE(
        base_channels=cfg.model.base_channels,
        channel_mults=tuple(cfg.model.channel_mults),
        strides=tuple(cfg.model.strides),
        latent_dim=cfg.model.latent_dim,
    ).to(device)
    stft_loss = MultiResSTFTLoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

    step = 0
    pbar = tqdm(total=cfg.train.max_steps, desc="stage5")
    while step < cfg.train.max_steps:
        for batch in loader:
            batch = batch.to(device)
            optim.zero_grad(set_to_none=True)
            rec, z = model(batch)
            l_time = F.l1_loss(rec, batch)
            l_stft = stft_loss(rec, batch)
            loss = cfg.loss.lambda_time * l_time + cfg.loss.lambda_stft * l_stft
            loss.backward()
            if cfg.train.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optim.step()
            step += 1
            pbar.update(1)

            if step % cfg.train.log_every == 0:
                logger.info(
                    f"step={step} loss={loss.item():.4f} time={l_time.item():.4f} stft={l_stft.item():.4f} "
                    f"latent_shape={tuple(z.shape)} downsample={model.downsample_ratio}"
                )

            if step % cfg.train.sample_every == 0:
                with torch.no_grad():
                    b0 = batch[0:1]
                    r0, _ = model(b0)
                    save_waveform(b0[0, 0], run_dir / f"gt_{step:06d}", sample_rate=cfg.data.sample_rate)
                    save_waveform(r0[0, 0], run_dir / f"rec_{step:06d}", sample_rate=cfg.data.sample_rate)
                    save_mel_spectrogram(
                        r0[0, 0], run_dir / f"rec_mel_{step:06d}", sample_rate=cfg.data.sample_rate,
                    )

            if step >= cfg.train.max_steps:
                break
    pbar.close()

    save_checkpoint(run_dir / "model.pt", {"model": model.state_dict(), "cfg": dict(cfg)})
    logger.info(f"saved {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
