"""Stage 3: rectified flow / CFM on MNIST with an Euler ODE solver."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_audiodit.data import build_mnist_loaders
from mini_audiodit.models.unet_2d import UNet2D
from mini_audiodit.paths import RectifiedFlow
from mini_audiodit.solvers import EulerSolver
from mini_audiodit.trainer import BaseTrainer
from mini_audiodit.utils import (
    load_config,
    save_checkpoint,
    save_image_grid,
    set_seed,
    setup_logger,
)


class RFTrainer(BaseTrainer):
    def __init__(self, sample_num: int, nfe_list: list[int], **kwargs) -> None:
        super().__init__(**kwargs)
        self.sample_num = sample_num
        self.nfe_list = nfe_list

    def training_step(self, batch):
        x, _ = batch
        z1 = x * 2 - 1
        z0 = RectifiedFlow.draw_prior(z1.shape, device=z1.device)
        t = RectifiedFlow.sample_t(z1.shape[0], device=z1.device)
        z_t = RectifiedFlow.interpolate(z0, z1, t)
        v_target = RectifiedFlow.target_velocity(z0, z1)
        v_pred = self.model(z_t, t)
        loss = F.mse_loss(v_pred, v_target)
        return loss, {"loss": loss.item()}

    def run_sampling(self) -> None:
        self.model.eval()
        for nfe in self.nfe_list:
            solver = EulerSolver(num_steps=nfe)
            shape = (self.sample_num, 1, 28, 28)
            z0 = RectifiedFlow.draw_prior(shape, device=self.device)
            samples = solver.integrate(
                velocity_fn=lambda z, t: self.model(z, t),
                z0=z0,
            )
            samples = (samples.clamp(-1, 1) + 1) / 2
            save_image_grid(
                samples, self.run_dir / f"samples_step{self.step:06d}_nfe{nfe}.png", nrow=8,
            )
        self.model.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    run_dir = ROOT / cfg.run_dir
    logger = setup_logger("stage3-rf", run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = build_mnist_loaders(
        ROOT / cfg.data.root, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
    )
    model = UNet2D(
        in_channels=1, out_channels=1,
        base_channels=cfg.model.base_channels, time_dim=cfg.model.time_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

    trainer = RFTrainer(
        sample_num=cfg.sample.num,
        nfe_list=list(cfg.sample.nfe_list),
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
