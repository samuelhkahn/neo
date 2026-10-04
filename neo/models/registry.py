"""Build the generator named by a training config's [MODEL] section (default: the NEO U-Net)."""

import ast

from neo.models.generator import Pix2PixGenerator


def generator_name(config) -> str:
    return config.get("MODEL", "generator", fallback="neo")


def _swinir(config):
    from neo.models.swinir import SwinIR

    section = config["SWINIR"] if config.has_section("SWINIR") else {}

    def get(key, default):
        return ast.literal_eval(section[key]) if key in section else default

    return SwinIR(
        embed_dim=get("embed_dim", 180),
        depths=tuple(get("depths", (6, 6, 6, 6, 6, 6))),
        num_heads=tuple(get("num_heads", (6, 6, 6, 6, 6, 6))),
        window_size=get("window_size", 8),
        mlp_ratio=get("mlp_ratio", 2.0),
        num_feat=get("num_feat", 64),
        use_checkpoint=get("use_checkpoint", False),
    )


def make_generator(config):
    name = generator_name(config)
    if name == "neo":
        return Pix2PixGenerator(1, 1)
    if name == "swinir":
        return _swinir(config)
    raise ValueError(f"unknown generator {name!r}")
