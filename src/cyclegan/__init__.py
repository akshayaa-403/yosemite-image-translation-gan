"""CycleGAN for unpaired summer <-> winter translation on Yosemite photos.

Domain A is summer, domain B is winter throughout. ``g_ab`` translates summer
into winter, ``g_ba`` does the reverse.
"""

from .buffer import ImageBuffer
from .config import Config
from .data import UnpairedImageDataset, build_dataloaders, resolve_split_dirs
from .discriminator import PatchDiscriminator
from .generator import ResidualBlock, ResnetGenerator
from .inference import load_generator, translate_image
from .trainer import CycleGANTrainer

__version__ = "0.2.0"

__all__ = [
    "Config",
    "CycleGANTrainer",
    "ImageBuffer",
    "PatchDiscriminator",
    "ResidualBlock",
    "ResnetGenerator",
    "UnpairedImageDataset",
    "build_dataloaders",
    "load_generator",
    "resolve_split_dirs",
    "translate_image",
]
