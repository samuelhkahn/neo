"""Build the generator named by a training config's [MODEL] section (default: the NEO U-Net)."""

from neo.models.generator import Pix2PixGenerator


def generator_name(config) -> str:
    return config.get("MODEL", "generator", fallback="neo")


def make_generator(config):
    name = generator_name(config)
    if name == "neo":
        return Pix2PixGenerator(1, 1)
    raise ValueError(f"unknown generator {name!r}")
