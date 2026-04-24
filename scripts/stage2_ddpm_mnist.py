"""Stage 2: DDPM baseline on MNIST -- the pedagogical foil for Flow Matching."""
from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_audiodit.data import build_mnist_loaders
from mini_audiodit.models.unet_2d import UNet2D
from mini_audiodit.paths import DDPMSchedule
from mini_audiodit.trainer import BaseTrainer
from mini_audiodit.utils import (
    load_config,
    save_checkpoint,
    save_image_grid,
    set_seed,
    setup_logger,
)


class DDPMTrainer(BaseTrainer):
    def __init__(self, schedule: DDPMSchedule, sample_num: int, sample_steps: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.schedule = schedule.to(self.device)
        self.sample_num = sample_num
        self.sample_steps = sample_steps

    def _model(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.model(x, t)

    def training_step(self, batch):
        x, _ = batch
        x = x * 2 - 1
        b = x.shape[0]
        t = torch.randint(0, self.schedule.num_steps, (b,), device=self.device)
        x_t, noise = self.schedule.q_sample(x, t)
        pred_eps = self._model(x_t, t.float())
        loss = F.mse_loss(pred_eps, noise)
        return loss, {"loss": loss.item()}

    def run_sampling(self) -> None:
        self.model.eval()
        shape = (self.sample_num, 1, 28, 28)
        samples = self.schedule.ancestral_sample(
            partial(self._model),
            shape,
            device=self.device,
            num_steps=self.sample_steps,
        )
        samples = (samples.clamp(-1, 1) + 1) / 2
        save_image_grid(samples, self.run_dir / f"samples_{self.step:06d}.png", nrow=8)
        self.model.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    run_dir = ROOT / cfg.run_dir
    logger = setup_logger("stage2-ddpm", run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = build_mnist_loaders(
        ROOT / cfg.data.root, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
    )
    model = UNet2D(
        in_channels=1, out_channels=1,
        base_channels=cfg.model.base_channels, time_dim=cfg.model.time_dim,
    )
    schedule = DDPMSchedule(
        num_steps=cfg.schedule.num_steps,
        beta_start=cfg.schedule.beta_start,
        beta_end=cfg.schedule.beta_end,
        schedule=cfg.schedule.type,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

    trainer = DDPMTrainer(
        schedule=schedule,
        sample_num=cfg.sample.num,
        sample_steps=cfg.sample.sample_steps,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
        logger=logger,
        max_steps=cfg.train.max_steps,
        log_every=cfg.train.log_every,
        sample_every=cfg.train.sample_every,
        val_every=cfg.train.val_every,
        grad_clip=cfg.train.grad_clip,
    )
    trainer.fit()
    save_checkpoint(run_dir / "model.pt", {"model": model.state_dict(), "cfg": dict(cfg)})


if __name__ == "__main__":
    main()
