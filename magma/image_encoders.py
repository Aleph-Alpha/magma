import torch
import torch.nn as nn
from typing import Callable, Union
from torchtyping import patch_typeguard
from einops import rearrange
import timm
import clip
from functools import partial

# ----------------------------- Utils --------------------------------------

clip.model.LayerNorm = (
    nn.LayerNorm
)  # we need to patch this for clip to work with deepspeed
patch_typeguard()  # needed for torchtyping typechecks to work


class Lambda(torch.nn.Module):
    def __init__(self, fn: Callable):
        super().__init__()
        assert hasattr(fn, "__call__")
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


# ------------------------- Image encoders ----------------------------------


def nfresnet50(
    device: Union[torch.device, str] = None, pretrained: bool = True
) -> nn.Module:
    """
    Loads nfresnet50 model, removing the pooling layer and replacing it with
    an adaptive pooling layer.
    """
    encoder = torch.nn.Sequential(
        *list(timm.create_model("nf_resnet50", pretrained=pretrained).children())[:-1]
    )
    pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
    encoder = torch.nn.Sequential(encoder, pooling)
    if device is not None:
        encoder = encoder.to(device)
    return encoder


def clip_encoder(
    device: Union[torch.device, str] = None, name: str = "clip",
) -> nn.Module:
    """
    Loads clip's image encoder module, discarding the lm component.

    If the variant is a resnet model, we also remove the attention pooling.
    """
    if name in ["clip", "ViT-B/32"]:
        name = "ViT-B/32"
    elif name in ["clip_resnet", "RN50x4"]:
        name = "RN50x4"
    elif name in ["clip_resnet_large", "RN50x16"]:
        name = "RN50x16"
    else:
        raise ValueError(f"encoder {name} not recognized")

    encoder = clip.load(name, device=device)[0].visual

    if device is not None:
        encoder = encoder.to(device)

    if "RN" in name:
        # remove attention pooling
        encoder.attnpool = Lambda(
            partial(rearrange, pattern="b d h w -> b (h w) d")
        )  # remove attn pooling, just use reshaped features

    return encoder


def get_image_encoder(
    name: str, device: Union[torch.device, str] = None, pretrained: bool = False
) -> torch.nn.Module:
    """
    Loads image encoder module
    """
    if name == "nfresnet50":
        encoder = nfresnet50(device=device, pretrained=pretrained)
    elif "clip" in name:
        encoder = clip_encoder(device=device, name=name)
    else:
        raise ValueError(f"image encoder {name} not recognized")
    return encoder
