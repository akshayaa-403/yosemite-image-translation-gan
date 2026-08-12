"""Training loop.

Structure of one step, in the order the reference implementation uses:

1. Generators: adversarial + cycle + identity, one backward through both.
2. Discriminator A and B: real vs. buffered fake, on *detached* fakes.

Three bugs from the original version of this project are fixed here and are
worth calling out, because each one silently degrades results rather than
crashing:

* ``d_y_optimizer.zero_grad()`` was never called, so gradients accumulated
  across every step for one of the two discriminators.
* the loop counted *iterations* as "epochs" and pulled one batch per epoch,
  re-creating the iterator on a modulo that did not line up with the loader
  length -- a ``StopIteration`` waiting to happen and no real epoch semantics.
* the learning rate was constant for the whole run; CycleGAN relies on linear
  decay to zero over the second half to stop late-training oscillation.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .buffer import ImageBuffer
from .config import Config
from .discriminator import PatchDiscriminator
from .generator import ResnetGenerator
from .losses import cycle_loss, discriminator_loss, gan_loss_real, identity_loss
from .metrics import cycle_metrics
from .utils import (
    CSVLogger,
    init_weights,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_image_grid,
    set_seed,
)

LOG_FIELDS = (
    "epoch",
    "step",
    "lr",
    "loss_g",
    "loss_g_adv",
    "loss_cycle",
    "loss_identity",
    "loss_d_a",
    "loss_d_b",
    "seconds",
)


class CycleGANTrainer:
    """Owns the four networks, their optimisers, and the run directory.

    Args:
        config: Resolved run configuration.
        train_loader: Yields dicts with ``"A"`` and ``"B"`` image batches.
        test_loader: Held-out data, used for sample grids and cycle metrics.
    """

    def __init__(self, config: Config, train_loader: DataLoader, test_loader: DataLoader):
        self.cfg = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        set_seed(config.seed)
        self.device = resolve_device(config.device)

        self.output_dir = Path(config.output_dir)
        self.sample_dir = self.output_dir / "samples"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        for directory in (self.sample_dir, self.checkpoint_dir):
            directory.mkdir(parents=True, exist_ok=True)
        config.to_yaml(self.output_dir / "config.yaml")

        # A: summer, B: winter.
        self.g_ab = self._make_generator()
        self.g_ba = self._make_generator()
        self.d_a = self._make_discriminator()
        self.d_b = self._make_discriminator()

        self.opt_g = torch.optim.Adam(
            itertools.chain(self.g_ab.parameters(), self.g_ba.parameters()),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )
        self.opt_d = torch.optim.Adam(
            itertools.chain(self.d_a.parameters(), self.d_b.parameters()),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )
        self.sched_g = self._make_scheduler(self.opt_g)
        self.sched_d = self._make_scheduler(self.opt_d)

        self.buffer_a = ImageBuffer(config.buffer_capacity, seed=config.seed)
        self.buffer_b = ImageBuffer(config.buffer_capacity, seed=config.seed + 1)

        # AMP only pays off on CUDA; on CPU it is a slowdown and a no-op.
        self.amp_enabled = bool(config.amp and self.device.type == "cuda")
        self.scaler_g = torch.amp.GradScaler(enabled=self.amp_enabled)
        self.scaler_d = torch.amp.GradScaler(enabled=self.amp_enabled)

        self.start_epoch = 1
        self.global_step = 0
        if config.resume:
            self._load(config.resume)

    # -- construction -------------------------------------------------------
    def _make_generator(self) -> ResnetGenerator:
        model = ResnetGenerator(
            base_channels=self.cfg.g_base_channels, n_res_blocks=self.cfg.n_res_blocks
        )
        model.apply(init_weights)
        return model.to(self.device)

    def _make_discriminator(self) -> PatchDiscriminator:
        model = PatchDiscriminator(base_channels=self.cfg.d_base_channels)
        model.apply(init_weights)
        return model.to(self.device)

    def _make_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
        """Constant LR until ``decay_start_epoch``, then linear decay to ~0."""
        decay_start = self.cfg.decay_start_epoch or max(1, self.cfg.epochs // 2)
        decay_epochs = max(1, self.cfg.epochs - decay_start + 1)

        def factor(epoch_index: int) -> float:
            # epoch_index is 0-based and advances once per scheduler.step(), so
            # epoch e (1-based) trains at factor(e - 1). The +2 makes decay
            # start *during* decay_start_epoch rather than the epoch after it.
            overshoot = epoch_index + 2 - decay_start
            if overshoot <= 0:
                return 1.0
            return max(0.0, 1.0 - overshoot / (decay_epochs + 1))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)

    # -- training -----------------------------------------------------------
    def train(self) -> None:
        """Run the configured number of epochs, sampling and checkpointing."""
        cfg = self.cfg
        print(
            f"Training on {self.device} | {len(self.train_loader)} steps/epoch "
            f"| batch {cfg.batch_size} | {cfg.crop_size}px | AMP {'on' if self.amp_enabled else 'off'}"
        )
        logger = CSVLogger(self.output_dir / "losses.csv", LOG_FIELDS)
        try:
            for epoch in range(self.start_epoch, cfg.epochs + 1):
                self._train_one_epoch(epoch, logger)
                self.sched_g.step()
                self.sched_d.step()

                if cfg.sample_every and epoch % cfg.sample_every == 0:
                    self.save_samples(epoch)
                if cfg.checkpoint_every and epoch % cfg.checkpoint_every == 0:
                    self.save(self.checkpoint_dir / f"epoch_{epoch:04d}.pt", epoch)
                self.save(self.checkpoint_dir / "latest.pt", epoch)
        finally:
            logger.close()
        print(f"Done. Artifacts in {self.output_dir}")

    def _train_one_epoch(self, epoch: int, logger: CSVLogger) -> None:
        self._set_train(True)
        started = time.time()
        totals = {"loss_g": 0.0, "loss_d_a": 0.0, "loss_d_b": 0.0}

        for step, batch in enumerate(self.train_loader, start=1):
            real_a = batch["A"].to(self.device, non_blocking=True)
            real_b = batch["B"].to(self.device, non_blocking=True)

            g_losses, fake_a, fake_b = self._generator_step(real_a, real_b)
            d_losses = self._discriminator_step(real_a, real_b, fake_a, fake_b)

            self.global_step += 1
            totals["loss_g"] += g_losses["loss_g"]
            totals["loss_d_a"] += d_losses["loss_d_a"]
            totals["loss_d_b"] += d_losses["loss_d_b"]

            if self.cfg.log_every and step % self.cfg.log_every == 0:
                row = {
                    "epoch": epoch,
                    "step": self.global_step,
                    "lr": self.opt_g.param_groups[0]["lr"],
                    "seconds": round(time.time() - started, 1),
                    **{k: round(v, 4) for k, v in {**g_losses, **d_losses}.items()},
                }
                logger.log(row)
                print(
                    f"epoch {epoch:>4}/{self.cfg.epochs} step {step:>5}/{len(self.train_loader)} "
                    f"| G {g_losses['loss_g']:.3f} (adv {g_losses['loss_g_adv']:.3f} "
                    f"cyc {g_losses['loss_cycle']:.3f} id {g_losses['loss_identity']:.3f}) "
                    f"| D_A {d_losses['loss_d_a']:.3f} D_B {d_losses['loss_d_b']:.3f}"
                )

        n = max(1, len(self.train_loader))
        print(
            f"epoch {epoch} done in {time.time() - started:.0f}s | "
            + " ".join(f"{k}={v / n:.3f}" for k, v in totals.items())
        )

    def _generator_step(
        self, real_a: torch.Tensor, real_b: torch.Tensor
    ) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        self.opt_g.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            fake_b = self.g_ab(real_a)
            fake_a = self.g_ba(real_b)
            rec_a = self.g_ba(fake_b)
            rec_b = self.g_ab(fake_a)

            # Fool both discriminators.
            adv = gan_loss_real(self.d_b(fake_b)) + gan_loss_real(self.d_a(fake_a))

            cyc = cycle_loss(real_a, rec_a) + cycle_loss(real_b, rec_b)

            if cfg.lambda_identity > 0:
                idt = identity_loss(real_a, self.g_ba(real_a)) + identity_loss(
                    real_b, self.g_ab(real_b)
                )
            else:
                idt = torch.zeros((), device=self.device)

            total = (
                adv
                + cfg.lambda_cycle * cyc
                + cfg.lambda_cycle * cfg.lambda_identity * idt
            )

        self.scaler_g.scale(total).backward()
        self.scaler_g.step(self.opt_g)
        self.scaler_g.update()

        losses = {
            "loss_g": total.item(),
            "loss_g_adv": adv.item(),
            "loss_cycle": cyc.item(),
            "loss_identity": float(idt.item()),
        }
        return losses, fake_a.detach(), fake_b.detach()

    def _discriminator_step(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor,
    ) -> dict[str, float]:
        # Both discriminators share one optimiser, so one zero_grad covers them
        # -- the bug being avoided is zeroing neither, not zeroing once.
        self.opt_d.zero_grad(set_to_none=True)

        # Mix in older fakes so D is not chasing only the newest generator state.
        buffered_a = self.buffer_a.push_and_sample(fake_a)
        buffered_b = self.buffer_b.push_and_sample(fake_b)

        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            loss_d_a = discriminator_loss(self.d_a(real_a), self.d_a(buffered_a))
            loss_d_b = discriminator_loss(self.d_b(real_b), self.d_b(buffered_b))
            total = loss_d_a + loss_d_b

        self.scaler_d.scale(total).backward()
        self.scaler_d.step(self.opt_d)
        self.scaler_d.update()

        return {"loss_d_a": loss_d_a.item(), "loss_d_b": loss_d_b.item()}

    # -- evaluation ---------------------------------------------------------
    @torch.no_grad()
    def save_samples(self, epoch: int, max_images: int = 4) -> Path:
        """Write a grid of held-out translations: real, fake, reconstruction."""
        self._set_train(False)
        rows_a, rows_b = [], []
        for batch in self.test_loader:
            if len(rows_a) >= max_images:
                break
            real_a = batch["A"].to(self.device)
            real_b = batch["B"].to(self.device)
            fake_b = self.g_ab(real_a)
            fake_a = self.g_ba(real_b)
            rows_a.append(torch.cat([real_a, fake_b, self.g_ba(fake_b)], dim=0).cpu())
            rows_b.append(torch.cat([real_b, fake_a, self.g_ab(fake_a)], dim=0).cpu())

        path = self.sample_dir / f"epoch_{epoch:04d}.png"
        # Row 1: summer -> winter -> back. Row 2: winter -> summer -> back.
        save_image_grid([torch.cat(rows_a, 0), torch.cat(rows_b, 0)], path, nrow=3)
        self._set_train(True)
        return path

    @torch.no_grad()
    def evaluate_cycle_consistency(self, max_batches: int | None = None) -> dict[str, float]:
        """Average cycle-reconstruction metrics over the test set."""
        self._set_train(False)
        sums: dict[str, float] = {}
        count = 0
        for i, batch in enumerate(self.test_loader):
            if max_batches is not None and i >= max_batches:
                break
            real_a = batch["A"].to(self.device)
            real_b = batch["B"].to(self.device)
            rec_a = self.g_ba(self.g_ab(real_a))
            rec_b = self.g_ab(self.g_ba(real_b))
            for prefix, real, rec in (("a", real_a, rec_a), ("b", real_b, rec_b)):
                for key, value in cycle_metrics(real, rec).items():
                    sums[f"{prefix}_{key}"] = sums.get(f"{prefix}_{key}", 0.0) + value
            count += 1
        self._set_train(True)
        if count == 0:
            return {}
        return {k: v / count for k, v in sums.items()}

    # -- persistence --------------------------------------------------------
    def _set_train(self, mode: bool) -> None:
        for model in (self.g_ab, self.g_ba, self.d_a, self.d_b):
            model.train(mode)

    def save(self, path: str | Path, epoch: int) -> None:
        save_checkpoint(
            path,
            epoch=epoch,
            global_step=self.global_step,
            g_ab=self.g_ab.state_dict(),
            g_ba=self.g_ba.state_dict(),
            d_a=self.d_a.state_dict(),
            d_b=self.d_b.state_dict(),
            opt_g=self.opt_g.state_dict(),
            opt_d=self.opt_d.state_dict(),
            sched_g=self.sched_g.state_dict(),
            sched_d=self.sched_d.state_dict(),
            config=vars(self.cfg).copy(),
        )

    def _load(self, path: str | Path) -> None:
        state: dict[str, Any] = load_checkpoint(path, map_location=self.device)
        self.g_ab.load_state_dict(state["g_ab"])
        self.g_ba.load_state_dict(state["g_ba"])
        self.d_a.load_state_dict(state["d_a"])
        self.d_b.load_state_dict(state["d_b"])
        self.opt_g.load_state_dict(state["opt_g"])
        self.opt_d.load_state_dict(state["opt_d"])
        self.sched_g.load_state_dict(state["sched_g"])
        self.sched_d.load_state_dict(state["sched_d"])
        self.start_epoch = int(state.get("epoch", 0)) + 1
        self.global_step = int(state.get("global_step", 0))
        print(f"Resumed from {path} at epoch {self.start_epoch}")
