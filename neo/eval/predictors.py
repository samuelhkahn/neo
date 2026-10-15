"""Load a trained model from its training config + checkpoint as a uniform predictor.

A predictor maps (lr, cond) -> output, all in the training log space:
  lr   (B, 1, 128, 128)  padded LR cutout, exactly as the dataset feeds the generator
  cond (B, 1, 768, 768)  6x nearest-neighbour upsampled LR (the dataset's `hsc_hr`)
  out  (B, 1, 768, 768)  predicted HR image
GAN generators use only `lr`; conditional samplers (e.g. diffusion) use `cond`.

A model trained with [NOISE] (neo/noise.py) predicts a clean image, which is the output by
default; noise="hst" adds an HST-like noise realisation at the config's default sky sigma (for
visual and realism comparisons with the noisy targets).
"""

import torch

from neo.models.registry import check_generator, generator_name, make_generator
from neo.noise import NoiseModel

NOISE_OPTIONS = ("none", "hst")


def noise_model(config, noise, device):
    """The NoiseModel for noise="hst" (None for "none"); needs a [NOISE] section."""
    if noise not in NOISE_OPTIONS:
        raise ValueError(f"noise must be one of {NOISE_OPTIONS}, not {noise!r}")
    if noise == "none":
        return None
    if not config.has_section("NOISE"):
        raise ValueError(
            "--noise hst needs a config with a [NOISE] section (its template and sigma)"
        )
    # read the run's cached template (or [NOISE] template), never measure one from local pairs
    return NoiseModel.from_config(config, device, measure=False)


def gan_predictor(config, checkpoint, device, mode: str = "eval", noise: str = "none"):
    gen = make_generator(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    check_generator(state, config, checkpoint)  # neo / neo_relu weights would load either way
    gen.load_state_dict(state["gen"])
    gen.to(device)
    gen.train(mode == "train")
    model = noise_model(config, noise, device)

    @torch.no_grad()
    def predict(lr, cond):
        out = gen(lr.to(device))
        if model is not None:
            out = model.add_noise(out, model.sigma_default)
        return out

    return predict


PREDICTORS = {"neo": gan_predictor}


def build_predictor(config, checkpoint, device, **options):
    name = generator_name(config)
    builder = PREDICTORS.get(name, gan_predictor)
    return builder(config, checkpoint, device, **options)
