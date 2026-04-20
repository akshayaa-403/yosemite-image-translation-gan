"""Loss functions module"""
from .losses import real_mse_loss, fake_mse_loss, cycle_consistency_loss

__all__ = ["real_mse_loss", "fake_mse_loss", "cycle_consistency_loss"]
