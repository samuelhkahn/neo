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


def check_generator(state, config, source="checkpoint") -> None:
    """Refuse a checkpoint written by another generator than the one this config builds.

    neo and neo_relu (Arm A) have the same state_dict keys, so a strict load_state_dict accepts
    either's weights and would silently run them through the other forward (a linear or a ReLU
    decoder). train.py therefore records the generator in every checkpoint ("generator"). One
    without that record was written before it existed: with the neo / neo_relu layout it is a
    parity (neo) run, since Arm A always records itself; for other generators (neo_lrtrunk,
    swinir), whose keys differ, the strict load itself catches a mismatch.
    """
    name = generator_name(config)
    saved = state.get("generator")
    if saved is None and name in ("neo", "neo_relu"):
        saved = "neo"
    if saved is not None and saved != name:
        raise ValueError(
            f"{source} was trained with generator {saved!r}, but the config builds {name!r} "
            "([MODEL] generator): use the run's own config"
        )
