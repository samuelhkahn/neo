"""Build the generator named by a training config's [MODEL] section (default: the NEO U-Net)."""

from neo.models.generator import Pix2PixGenerator


def generator_name(config) -> str:
    return config.get("MODEL", "generator", fallback="neo")


def make_generator(config):
    name = generator_name(config)
    # Ablations of the NEO U-Net (neo/models/generator_variants.py): Arm A turns on the decoder
    # ReLU the paper specifies, Arm B adds a trunk on the 128 px LR grid to Arm A
    if name == "neo_relu":
        from neo.models.generator_variants import NeoReLUGenerator

        return NeoReLUGenerator(1, 1)
    if name == "neo_lrtrunk":
        from neo.models.generator_variants import NeoLRTrunkGenerator

        return NeoLRTrunkGenerator(1, 1)
    if name == "neo":
        return Pix2PixGenerator(1, 1)
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
