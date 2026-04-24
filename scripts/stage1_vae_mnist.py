"""Stage 1: train a plain VAE on MNIST."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_audiodit.data import build_mnist_loaders
from mini_audiodit.models.vae_mnist import VAE, vae_loss
from mini_audiodit.trainer import BaseTrainer
from mini_audiodit.utils import (
    load_config,
    plot_latent_histogram,
    save_checkpoint,
    save_image_grid,
    set_seed,
    setup_logger,
)


class VAETrainer(BaseTrainer):
    def __init__(self, beta: float, sample_num: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = beta
        self.sample_num = sample_num

    def training_step(self, batch):
        x, _ = batch
        logits, mu, logvar, _ = self.model(x)
        loss, metrics = vae_loss(logits, x, mu, logvar, beta=self.beta)
        return loss, metrics

    def run_sampling(self) -> None:
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(self.sample_num, device=self.device)
            save_image_grid(samples, self.run_dir / f"samples_{self.step:06d}.png", nrow=8)

            val_batch = next(iter(self.val_loader))[0][:256].to(self.device)
            mu, _ = self.model.encode(val_batch)
            plot_latent_histogram(mu, self.run_dir / f"latent_hist_{self.step:06d}.png")
        self.model.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    run_dir = ROOT / cfg.run_dir
    logger = setup_logger("stage1-vae", run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = build_mnist_loaders(
        ROOT / cfg.data.root, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
    )
    model = VAE(hidden_dim=cfg.model.hidden_dim, latent_dim=cfg.model.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)

    trainer = VAETrainer(
        beta=cfg.train.beta,
        sample_num=cfg.sample.num,
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
