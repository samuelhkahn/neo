"""Load a trained model from its training config + checkpoint as a uniform predictor.

A predictor maps (lr, cond) -> output, all in the training log space:
  lr   (B, 1, 128, 128)  padded LR cutout, exactly as the dataset feeds the generator
  cond (B, 1, 768, 768)  6x nearest-neighbour upsampled LR (the dataset's `hsc_hr`)
  out  (B, 1, 768, 768)  predicted HR image
GAN generators use only `lr`; conditional samplers (e.g. diffusion) use `cond`.
"""

import torch

from neo.models.registry import generator_name, make_generator


def gan_predictor(config, checkpoint, device, mode: str = "eval"):
    gen = make_generator(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    gen.load_state_dict(state["gen"])
    gen.to(device)
    gen.train(mode == "train")

    @torch.no_grad()
    def predict(lr, cond):
        return gen(lr.to(device))

    return predict


PREDICTORS = {"neo": gan_predictor}


def build_predictor(config, checkpoint, device, **options):
    name = generator_name(config)
    builder = PREDICTORS.get(name, gan_predictor)
    return builder(config, checkpoint, device, **options)
