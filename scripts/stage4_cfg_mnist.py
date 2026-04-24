"""Stage 4: conditional rectified flow + classifier-free guidance on MNIST."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_audiodit.data import build_mnist_loaders
from mini_audiodit.guidance import cfg_velocity
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


class CFGTrainer(BaseTrainer):
    def __init__(self, num_classes: int, num_per_class: int, alphas: list[float], nfe: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_per_class = num_per_class
        self.alphas = alphas
        self.nfe = nfe

    def training_step(self, batch):
        x, y = batch
        z1 = x * 2 - 1
        z0 = RectifiedFlow.draw_prior(z1.shape, device=z1.device)
        t = RectifiedFlow.sample_t(z1.shape[0], device=z1.device)
        z_t = RectifiedFlow.interpolate(z0, z1, t)
        v_target = RectifiedFlow.target_velocity(z0, z1)
        v_pred = self.model(z_t, t, y)
        loss = F.mse_loss(v_pred, v_target)
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def _sample_grid(self, alpha: float) -> torch.Tensor:
        device = self.device
        labels = torch.arange(self.num_classes, device=device).repeat_interleave(self.num_per_class)
        n = labels.shape[0]
        z0 = torch.randn(n, 1, 28, 28, device=device)

        def vel(z, t):
            v_cond = self.model(z, t, labels)
            v_uncond = self.model(z, t, None)
            return cfg_velocity(v_cond, v_uncond, alpha=alpha)

        solver = EulerSolver(num_steps=self.nfe)
        z = solver.integrate(velocity_fn=vel, z0=z0)
        return (z.clamp(-1, 1) + 1) / 2

    def run_sampling(self) -> None:
        self.model.eval()
        for alpha in self.alphas:
            grid = self._sample_grid(alpha)
            save_image_grid(
                grid,
                self.run_dir / f"samples_step{self.step:06d}_alpha{alpha:.1f}.png",
                nrow=self.num_per_class,
            )
        self.model.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    run_dir = ROOT / cfg.run_dir
    logger = setup_logger("stage4-cfg", run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = build_mnist_loaders(
        ROOT / cfg.data.root, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
    )
    model = UNet2D(
        in_channels=1, out_channels=1,
        base_channels=cfg.model.base_channels, time_dim=cfg.model.time_dim,
        num_classes=cfg.model.num_classes,
        class_drop_prob=cfg.model.class_drop_prob,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

    trainer = CFGTrainer(
        num_classes=cfg.model.num_classes,
        num_per_class=cfg.sample.num_per_class,
        alphas=list(cfg.sample.alphas),
        nfe=cfg.sample.nfe,
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
