import torch
import torch.nn as nn
from torchtyping import TensorType
from einops import rearrange
from .image_encoders import get_image_encoder
from .config import MultimodalConfig

# ------------------------- Image prefix ----------------------------------

# for models that are fixed to a specific sequence lengths (i.e clip models with no pooling), the sequence lengths are below
ENCODER_SEQ_LENS = {
    "clip_resnet": 49,
    "clip_resnet_large": 144,
}

ENCODER_OUT_DIMS = {
    "nfresnet50": 2048,
    "clip": 512,
    "clip_resnet": 2560,
    "clip_resnet_large": 3072,
}


class ImagePrefix(nn.Module):

    """
    Takes in a batch of images and returns a batch of embeddings of the
    same dimensions as the LM's word embeddings.

    :param config: MultimodalConfig object
    :param out_dim: output dimension of the embedding
    :param device: device to run the model on
    """

    def __init__(
        self,
        config: MultimodalConfig,
        out_dim: int = 2048,
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = config
        self.encoder_type = config.encoder_name

        # get image encoder backbone
        self.enc = get_image_encoder(
            config.encoder_name,
            pretrained=config.pretrained_img_encoder,
        )
        self.encoder_out_dim = ENCODER_OUT_DIMS[
            self.encoder_type
        ]  # out dim for image encoder

        self.out_dim = out_dim  # out dim for lm

        # set the out seq len to that specified in the config, or for some models, the hardcoded value
        self.out_seq_len = (
            config.image_seq_len
            if config.encoder_name not in ENCODER_SEQ_LENS
            else ENCODER_SEQ_LENS[config.encoder_name]
        )

        # get the output projection
        proj_out_dim = (
            (self.out_dim * self.out_seq_len)
            if self.encoder_type not in ENCODER_SEQ_LENS
            else self.out_dim
        )
        self.proj = nn.Linear(self.encoder_out_dim, proj_out_dim)
        self.dropout = nn.Dropout(config.image_embed_dropout_prob)
        self.use_layernorm = config.use_image_embed_layernorm
        if self.use_layernorm:
            self.ln = nn.LayerNorm(self.out_dim)

    def forward(
        self, x: TensorType["b", "c", "h", "w"]
    ) -> TensorType["b", "seq", "out_dim"]:

        # pass through image encoder
        logits = self.enc(x)

        # remove trailing dimensions of size 1 + pass through linear
        if logits.ndim == 4:
            logits = rearrange(logits, "b d 1 1 -> b d")
        elif logits.ndim == 3:
            assert self.encoder_type in ENCODER_SEQ_LENS
        else:
            assert logits.ndim == 2

        logits = self.proj(logits)

        # reshape to desired output shape
        if (
            self.encoder_type not in ENCODER_SEQ_LENS
        ):  # don't need to reshape those with fixed seq lens / no pooling
            logits = rearrange(
                logits, "b (s d) -> b s d", d=self.out_dim, s=self.out_seq_len
            )

        # pass through dropout and layer norm
        logits = self.dropout(logits)

        if self.use_layernorm:
            logits = self.ln(logits)

        return logits
