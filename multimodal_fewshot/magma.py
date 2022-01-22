from path import Path
import torch
import torch.nn as nn
import re
from copy import deepcopy

from typing import Literal, Optional, List, Callable, Union
from torchtyping import TensorType
import transformers
from transformers.file_utils import ModelOutput
import torch.nn.functional as F
from multimodal_fewshot.config import MultimodalConfig

from multimodal_fewshot.utils import get_tokenizer, infer_checkpoint_path_from_config
from .sampling import top_k_filter, top_p_filter
from .language_model import get_language_model
from .adapters import (
    Adapter,
    ParallelAdapter,
    AdapterWrapper,
    ParallelAdapterWrapper,
)
from .image_prefix import ImagePrefix
from .sampling import generate
from .utils import build_labels
from .transforms import get_transforms

# ------------------------- Magma main class ----------------------------------


class Magma(nn.Module):
    def __init__(self, config, device=None, model_dir="./", lm_from_pretrained=False):
        super().__init__()

        if isinstance(config, (str, Path)):
            config = MultimodalConfig.from_yml(
                config
            )  # load config from yml file if config is a string
        else:
            assert isinstance(config, MultimodalConfig)

        self.config = config
        self.lm = get_language_model(
            config.lm_name, model_dir=model_dir, from_pretrained=lm_from_pretrained
        )
        self.seq_len = self.lm.config.max_position_embeddings

        self.tokenizer = get_tokenizer("gpt2", sequence_length=self.seq_len)

        # setup lm settings
        self.tokenizer.add_special_tokens(
            {"cls_token": "<|image|>"}
        )  # add special image token to tokenizer
        self.image_token = self.tokenizer.cls_token_id
        self.eos_token = self.tokenizer.eos_token_id
        self.lm.resize_token_embeddings(len(self.tokenizer))
        self.lm.config.pad_token_id = self.tokenizer.eos_token_id
        self.word_embedding = self.lm.transformer.wte
        self.transformer = self.lm.transformer.h

        # adapter settings
        self.mlp_adapter_added, self.attn_adapter_added = False, False

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.image_prefix = ImagePrefix(
            config=config, out_dim=self.lm.config.hidden_size,
        )

        # might change based on the type of image encoder, so get from prefix instead of config
        self.image_prefix_seq_len = self.image_prefix.out_seq_len

        self.transforms = get_transforms(
            config.image_size,
            config.encoder_name,
            input_resolution=self.image_prefix.enc.input_resolution,
        )

        # add adapters
        if config.adapter_config:
            mlp_config = deepcopy(config.adapter_config.get("mlp", None))
            if mlp_config:
                assert mlp_config.get("adapter_type") is not None
                self.add_adapters(
                    location="mlp",
                    adapter_type=mlp_config.pop("adapter_type"),
                    downsample_factor=mlp_config.pop("downsample_factor", 4),
                    **mlp_config,
                )
            attn_config = deepcopy(config.adapter_config.get("attention", None))
            if attn_config:
                assert attn_config.get("adapter_type") is not None
                self.add_adapters(
                    location="attention",
                    adapter_type=attn_config.pop("adapter_type"),
                    **attn_config,
                )

        # freeze parameters
        if config.freeze_lm:
            for name, param in self.lm.named_parameters():  # freeze lm weights
                if config.adapter_config and "adapter" in name:
                    param.requires_grad = True

        if config.freeze_img_encoder:
            for param in self.image_prefix.enc.parameters():
                param.requires_grad = False

    def add_adapters(
        self,
        downsample_factor: int = 4,
        adapter_type: Literal["normal", "parallel", "scaled_parallel"] = "normal",
        location: Literal["mlp", "attention"] = "mlp",
        ff_attr: str = "mlp",
        attn_attr: str = "attn",
        **adapter_kwargs,
    ):
        """
        Adds an adapter layer to `self` at the specified location
        """
        assert adapter_type in [
            "normal",
            "parallel",
            "scaled_parallel",
        ], "adapter_type must be one of 'normal', 'parallel', or 'scaled_parallel'"
        assert location in [
            "mlp",
            "attention",
        ], "location must be one of 'mlp' or 'attention'"

        for l in range(len(self.transformer)):
            if location == "mlp":
                if self.mlp_adapter_added:
                    raise ValueError("Adapter layer already added")
                mlp = getattr(self.transformer[l], ff_attr)
                if adapter_type in ["parallel", "scaled_parallel"]:
                    adapter_layer = ParallelAdapter(
                        module=mlp,
                        dim=self.lm.config.hidden_size,
                        downsample_factor=downsample_factor,
                        scaled=adapter_type == "scaled_parallel",
                        **adapter_kwargs,
                    )
                else:
                    adpt = Adapter(
                        dim=self.lm.config.hidden_size,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs,
                    )
                    adapter_layer = nn.Sequential(*[mlp, adpt,])
                setattr(self.transformer[l], ff_attr, adapter_layer)
            else:
                if self.attn_adapter_added:
                    raise ValueError("Adapter layer already added")
                attn = getattr(self.transformer[l], attn_attr)
                if adapter_type in ["parallel", "scaled_parallel"]:
                    adapter_layer = ParallelAdapterWrapper(
                        module=attn,
                        dim=self.lm.config.hidden_size,
                        downsample_factor=downsample_factor,
                        scaled="scaled" in adapter_type,
                        **adapter_kwargs,
                    )
                else:
                    adapter_layer = AdapterWrapper(
                        attn_block=attn,
                        dim=self.lm.config.hidden_size,
                        downsample_factor=downsample_factor,
                        **adapter_kwargs,
                    )
                setattr(self.transformer[l], attn_attr, adapter_layer)

        if location == "mlp":
            self.mlp_adapter_added = True
        else:
            self.attn_adapter_added = True

    def embed(self, inputs: List[torch.Tensor]) -> TensorType["b", "s", "d"]:
        """
        Embeds a list of tensors In the correct format to input into the LM (b, s, d).
        For each tensor, if it's 2d assume it's text and use word embedding,
        if it's 4d, assume it's an image, and use image_prefix to embed.
        """
        emb_list = []
        for x in inputs:
            if x.ndim == 2:
                x = x.to(self.device)
                emb_list.append(self.word_embedding(x))
            elif x.ndim == 4:
                x = x.to(self.device).half()
                image_embeddings = self.image_prefix(x)
                emb_list.append(image_embeddings)
            else:
                raise ValueError(f"Expected 2d or 4d tensor, got {x.ndim}d")
        return torch.cat(emb_list, dim=1)

    @torch.no_grad()
    def generate(
        self,
        embeddings: TensorType["b", "s", "d"],
        max_steps: int = 100,
        temperature: float = 0.7,
        top_k: int = 0,
        top_p: float = 0.9,
        decode: bool = True,
    ):
        """
        Generates captions for a batch of embeddings.
        """

        return generate(self, embeddings, max_steps, temperature, top_k, top_p, decode)

    def forward(
        self,
        images: TensorType["b", "c", "h", "w"] = None,
        captions: Optional[TensorType["b", "seq"]] = None,
        output_hidden_states: bool = False,
        input_embeddings: TensorType["b", "s", "d"] = None,
    ) -> ModelOutput:
        assert captions is not None, "Must provide captions in training"
        assert any([i is not None for i in [images, input_embeddings]]) and not all(
            [i is not None for i in [images, input_embeddings]]
        ), "Pass in either images, or input embeddings, not both."
        assert (
            captions.shape[1] == self.seq_len
        ), f"in training, captions should be padded to sequence length ({self.seq_len}), but are length {captions.shape[1]}"

        if input_embeddings is None:
            input_embeddings = self.image_prefix(images)
        labels = build_labels(
            input_embeddings, captions, self.eos_token, self.device
        )  # build labels from input_embeddings
        word_embeddings = self.word_embedding(captions)

        # join together
        input_embeddings = torch.cat(
            (
                input_embeddings,
                word_embeddings[:, : -input_embeddings.shape[1], :],
            ),  # remove padding in the word embedding before concatenating
            dim=1,
        )

        # forward joined embeddings through lm
        lm_outputs = self.lm(
            inputs_embeds=input_embeddings,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )

        return lm_outputs


def get_multimodal_model(
    config_path,
    model_dir="models",
    ckpt_path=None,
    tokenizer_name="gpt2",
    lm_from_pretrained=False,
):
    from .config import MultimodalConfig
    from .utils import get_tokenizer
    from .transforms import get_transforms

    tokenizer = get_tokenizer(tokenizer_name)
    config = MultimodalConfig.from_yml(config_path)

    model = Magma(
        lm=get_language_model(
            config.lm_name,
            model_dir=model_dir,
            from_pretrained=lm_from_pretrained,
            no_init=True,
        ),
        tokenizer=tokenizer,
        config=config,
    )

    transforms = get_transforms(config.image_size, model=model)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location=torch.device("cpu"))
        print("loading multimodal transformer checkpoint...")
        model.load_state_dict(sd["module"])
        print(f"loaded multimodal transformer from checkpoint {ckpt_path}")

    return model, transforms, tokenizer
