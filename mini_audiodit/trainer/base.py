"""A tiny generic trainer that all stages build on."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader


class BaseTrainer:
    """Minimal loop.  Subclasses override `training_step` and (optionally) `validation_step`."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: str | torch.device,
        run_dir: str | Path,
        logger: logging.Logger,
        max_steps: int,
        log_every: int = 50,
        sample_every: int = 500,
        val_every: int = 500,
        grad_clip: float | None = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(device)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.max_steps = max_steps
        self.log_every = log_every
        self.sample_every = sample_every
        self.val_every = val_every
        self.grad_clip = grad_clip
        self.step = 0

    def fit(self) -> None:
        self.model.train()
        done = False
        while not done:
            for batch in self.train_loader:
                batch = self._to_device(batch)
                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = self.training_step(batch)
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.step += 1

                if self.step % self.log_every == 0:
                    msg = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    self.logger.info(f"step={self.step:6d} {msg}")

                if self.val_loader is not None and self.step % self.val_every == 0:
                    self.run_validation()

                if self.step % self.sample_every == 0:
                    self.run_sampling()

                if self.step >= self.max_steps:
                    done = True
                    break

        self.run_sampling()
        self.logger.info("training finished.")

    def run_validation(self) -> None:
        if self.val_loader is None:
            return
        self.model.eval()
        totals: dict[str, float] = {}
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)
                metrics = self.validation_step(batch)
                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.0) + float(v)
                n += 1
        if n > 0:
            msg = " ".join(f"val/{k}={v / n:.4f}" for k, v in totals.items())
            self.logger.info(f"step={self.step:6d} {msg}")
        self.model.train()

    def training_step(self, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def validation_step(self, batch: Any) -> dict[str, float]:
        loss, metrics = self.training_step(batch)
        return metrics

    def run_sampling(self) -> None:
        pass

    def _to_device(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(x) for x in batch)
        if isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        return batch
