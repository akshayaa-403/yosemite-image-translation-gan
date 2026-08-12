"""End-to-end training behaviour on the synthetic dataset.

These run on CPU in seconds with a deliberately tiny model. They are here to
catch the class of bug that does not raise: gradients that never get zeroed,
schedulers that never decay, checkpoints that do not restore.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cyclegan.config import Config
from cyclegan.data import build_dataloaders
from cyclegan.trainer import CycleGANTrainer


def make_trainer(fake_dataset: Path, tmp_path: Path, **overrides) -> CycleGANTrainer:
    config = Config.from_dict(
        {
            "data_root": str(fake_dataset),
            "load_size": 72,
            "crop_size": 64,
            "batch_size": 2,
            "num_workers": 0,
            "g_base_channels": 8,
            "d_base_channels": 8,
            "n_res_blocks": 1,
            "epochs": 2,
            "buffer_capacity": 4,
            "amp": False,
            "device": "cpu",
            "output_dir": str(tmp_path / "run"),
            "log_every": 1,
            "sample_every": 1,
            "checkpoint_every": 1,
            **overrides,
        }
    )
    train_loader, test_loader = build_dataloaders(
        config.data_root, config.load_size, config.crop_size, config.batch_size, 0
    )
    return CycleGANTrainer(config, train_loader, test_loader)


def test_full_run_writes_samples_checkpoints_and_losses(fake_dataset: Path, tmp_path: Path) -> None:
    trainer = make_trainer(fake_dataset, tmp_path)
    trainer.train()

    run = Path(trainer.cfg.output_dir)
    assert (run / "config.yaml").is_file()
    assert (run / "losses.csv").is_file()
    assert sorted(p.name for p in (run / "samples").glob("*.png")) == [
        "epoch_0001.png",
        "epoch_0002.png",
    ]
    assert (run / "checkpoints" / "latest.pt").is_file()

    header, *rows = (run / "losses.csv").read_text(encoding="utf-8").strip().splitlines()
    assert "loss_d_a" in header and "loss_identity" in header
    assert rows, "at least one loss row should be logged"


def test_discriminator_gradients_do_not_accumulate_across_steps(
    fake_dataset: Path, tmp_path: Path
) -> None:
    """Regression test for the missing zero_grad on the second discriminator."""
    trainer = make_trainer(fake_dataset, tmp_path)
    batch = next(iter(trainer.train_loader))
    real_a, real_b = batch["A"], batch["B"]

    def grad_norm() -> float:
        params = [p for p in trainer.d_b.parameters() if p.grad is not None]
        return sum(p.grad.norm().item() for p in params)

    _, fake_a, fake_b = trainer._generator_step(real_a, real_b)
    trainer._discriminator_step(real_a, real_b, fake_a, fake_b)
    first = grad_norm()

    # Re-run the same step with frozen weights: if gradients were accumulating,
    # the second norm would be roughly double the first.
    for param in trainer.d_b.parameters():
        param.requires_grad_(True)
    trainer._discriminator_step(real_a, real_b, fake_a, fake_b)
    second = grad_norm()

    assert second < first * 1.6, f"gradients look accumulated: {first:.4f} then {second:.4f}"


def test_generator_weights_change_after_a_step(fake_dataset: Path, tmp_path: Path) -> None:
    trainer = make_trainer(fake_dataset, tmp_path)
    before = trainer.g_ab.model[1].weight.detach().clone()
    batch = next(iter(trainer.train_loader))
    trainer._generator_step(batch["A"], batch["B"])
    assert not torch.allclose(before, trainer.g_ab.model[1].weight)


def test_identity_loss_is_reported_as_zero_when_disabled(fake_dataset: Path, tmp_path: Path) -> None:
    trainer = make_trainer(fake_dataset, tmp_path, lambda_identity=0.0)
    batch = next(iter(trainer.train_loader))
    losses, _, _ = trainer._generator_step(batch["A"], batch["B"])
    assert losses["loss_identity"] == 0.0
    assert losses["loss_g"] > 0


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step\\(\\)`")
def test_learning_rate_decays_linearly_to_zero(fake_dataset: Path, tmp_path: Path) -> None:
    """Stepping the scheduler alone is fine here -- we only inspect the LR."""
    trainer = make_trainer(fake_dataset, tmp_path, epochs=4, decay_start_epoch=3)
    seen = []
    for _ in range(4):
        seen.append(trainer.opt_g.param_groups[0]["lr"])
        trainer.sched_g.step()

    assert seen[0] == seen[1] == trainer.cfg.lr, "constant before decay_start_epoch"
    assert seen[2] < seen[1] and seen[3] < seen[2], "monotonically decaying afterwards"
    assert trainer.opt_g.param_groups[0]["lr"] == 0.0, "reaches zero at the end of the run"


def test_resume_restores_weights_and_epoch(fake_dataset: Path, tmp_path: Path) -> None:
    trainer = make_trainer(fake_dataset, tmp_path)
    trainer.train()
    checkpoint = Path(trainer.cfg.output_dir) / "checkpoints" / "latest.pt"

    resumed = make_trainer(fake_dataset, tmp_path, resume=str(checkpoint), epochs=3)
    assert resumed.start_epoch == 3
    assert resumed.global_step == trainer.global_step
    original = trainer.g_ab.state_dict().values()
    restored = resumed.g_ab.state_dict().values()
    for a, b in zip(original, restored, strict=True):
        torch.testing.assert_close(a, b)


def test_cycle_metrics_are_finite_and_bounded(fake_dataset: Path, tmp_path: Path) -> None:
    trainer = make_trainer(fake_dataset, tmp_path)
    metrics = trainer.evaluate_cycle_consistency(max_batches=2)
    assert set(metrics) == {"a_l1", "a_psnr", "a_ssim", "b_l1", "b_psnr", "b_ssim"}
    assert 0.0 <= metrics["a_l1"] <= 1.0
    assert -1.0 <= metrics["a_ssim"] <= 1.0
    assert all(v == v for v in metrics.values()), "no NaNs"


def test_checkpoint_carries_the_config_for_later_loading(fake_dataset: Path, tmp_path: Path) -> None:
    """inference.load_generator rebuilds the architecture from this."""
    trainer = make_trainer(fake_dataset, tmp_path)
    trainer.save(Path(trainer.cfg.output_dir) / "checkpoints" / "manual.pt", epoch=1)
    state = torch.load(
        Path(trainer.cfg.output_dir) / "checkpoints" / "manual.pt", weights_only=True
    )
    assert state["config"]["n_res_blocks"] == 1
    assert state["config"]["g_base_channels"] == 8
    assert json.dumps(state["config"], default=str)  # plain data, no pickled objects
