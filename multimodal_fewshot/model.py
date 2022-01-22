import torch
import torch.nn as nn
import re
from copy import deepcopy

from typing import Literal, Optional, List, Callable, Union
from torchtyping import TensorType, patch_typeguard
import transformers
from transformers.file_utils import ModelOutput
from einops import rearrange
import torch.nn.functional as F
import timm
import clip
from functools import partial

from multimodal_fewshot.utils import get_tokenizer, infer_checkpoint_path_from_config
from .sampling import top_k, top_p
from .language_model import get_language_model
from .adapters import (
    Adapter,
    ParallelAdapter,
    AdapterWrapper,
    ParallelAdapterWrapper,
)

clip.model.LayerNorm = (
    nn.LayerNorm
)  # we need to patch this for it to work with deepspeed
patch_typeguard()  # needed for torchtyping typechecks to work


class Lambda(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        assert hasattr(fn, "__call__")
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


# ------------------------- Image encoder ----------------------------------


def get_image_encoder(
    name: str, device: Union[torch.device, str] = None, pretrained: bool = False
) -> torch.nn.Module:
    """
    Loads image encoder module
    """
    if name == "nfresnet50":
        encoder = torch.nn.Sequential(
            *list(timm.create_model("nf_resnet50", pretrained=True).children())[:-1]
        )
        pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        encoder = torch.nn.Sequential(encoder, pooling)
    elif name == "clip":
        model, _ = clip.load(
            "ViT-B/32", device="cpu"
        )  # this actually returns encoder, preprocess_fn
        encoder = model.visual
    elif name == "clip_no_pooling":
        model, _ = clip.load("ViT-B/32", device="cpu")
        encoder = model.visual

        # patch the ViT forward function to remove the pooling
        def forward_patch(self, x: torch.Tensor):
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [
                    self.class_embedding.to(x.dtype)
                    + torch.zeros(
                        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                    ),
                    x,
                ],
                dim=1,
            )  # shape = [*, grid ** 2 + 1, width]
            x = x + self.positional_embedding.to(x.dtype)
            x = self.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

            # removed: # TODO: keep layernorm?
            # x = self.ln_post(x[:, 0, :])
            # if self.proj is not None:
            #     x = x @ self.proj
            return x

        encoder.forward = forward_patch.__get__(
            encoder, clip.model.VisionTransformer
        )  # will output a [b, 50, 768] matrix
    elif name == "clip_resnet":
        model, _ = clip.load(
            "RN50x4", device="cpu"
        )  # this actually returns encoder, preprocess_fn
        encoder = model.visual
        encoder.attnpool = Lambda(
            partial(rearrange, pattern="b d h w -> b (h w) d")
        )  # remove attn pooling, just use features
    elif name == "clip_resnet_large":
        model, _ = clip.load(
            "RN50x16", device="cpu"
        )  # this actually returns encoder, preprocess_fn
        encoder = model.visual
        encoder.attnpool = Lambda(
            partial(rearrange, pattern="b d h w -> b (h w) d")
        )  # remove attn pooling, just use reshaped features
    else:
        raise ValueError(f"image encoder {name} not recognized")
    return encoder


# ------------------------- Image prefix ----------------------------------


class ImagePrefix(nn.Module):

    # for models that are fixed to a specific sequence lengths (i.e clip models with no pooling), the sequence lengths are below
    IMAGE_ENCODER_SEQ_LENS = {
        "clip_no_pooling": 50,  # 1 extra for [CLS] token
        "clip_resnet": 49,
        "clip_resnet_large": 144,
    }

    IMAGE_ENCODER_OUT_DIMS = {
        "nfresnet50": 2048,
        "clip": 512,
        "clip_no_pooling": 768,
        "clip_resnet": 2560,
        "clip_resnet_large": 3072,
    }

    def __init__(
        self, config, out_dim: int = 2048, device=None,
    ):
        super().__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = config
        self.image_encoder_type = config.encoder_name

        # get (maybe pretrained) image encoder (clip / resnet etc.)
        self.enc = get_image_encoder(
            config.encoder_name, pretrained=config.pretrained_img_encoder,
        )
        self.encoder_out_dim = self.IMAGE_ENCODER_OUT_DIMS[
            self.image_encoder_type
        ]  # out dim for image encoder
        self.out_dim = out_dim  # out dim for lm

        # set the out seq len to that specified in the config, or for some models, the hardcoded value
        self.out_seq_len = (
            config.image_seq_len
            if config.encoder_name not in self.IMAGE_ENCODER_SEQ_LENS
            else self.IMAGE_ENCODER_SEQ_LENS[config.encoder_name]
        )

        # get the output projection
        proj_out_dim = (
            (self.out_dim * self.out_seq_len)
            if self.image_encoder_type not in self.IMAGE_ENCODER_SEQ_LENS
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
            assert self.image_encoder_type in self.IMAGE_ENCODER_SEQ_LENS
        else:
            assert logits.ndim == 2
        logits = self.proj(logits)

        # reshape to desired output shape
        if (
            self.image_encoder_type not in self.IMAGE_ENCODER_SEQ_LENS
        ):  # don't need to reshape those with fixed seq lens / no pooling
            logits = rearrange(
                logits, "b (s d) -> b s d", d=self.out_dim, s=self.out_seq_len
            )

        # pass through dropout and layer norm
        logits = self.dropout(logits)
        if self.use_layernorm:
            logits = self.ln(logits)

        return logits


# ------------------------- Multimodal transformer main class ----------------------------------


class MultimodalLM(nn.Module):
    def __init__(
        self,
        lm: nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        config,
        device=None,
    ):
        super().__init__()
        self.lm = lm
        self.tokenizer = tokenizer
        self.config = config

        # setup lm stuff
        self.tokenizer.add_special_tokens(
            {"cls_token": "<|image|>"}
        )  # add special image token to tokenizer
        self.image_token = self.tokenizer.cls_token_id
        self.eos_token = self.tokenizer.eos_token_id
        self.lm.resize_token_embeddings(len(self.tokenizer))
        self.transformer = self.lm.transformer.h
        self.lm.config.pad_token_id = tokenizer.eos_token_id
        self.word_embedding = self.lm.transformer.wte
        self.mlp_adapter_added = False
        self.attn_adapter_added = False
        self.seq_len = self.lm.config.max_position_embeddings

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.image_prefix = ImagePrefix(
            config=config, out_dim=self.lm.config.hidden_size,
        )

        # might change based on the type of image encoder, so get from prefix instead of config
        self.image_prefix_seq_len = self.image_prefix.out_seq_len

        # add adapters
        if config.adapter_config:
            # adapter config should be a dict like so:
            # {
            #     "mlp": {"adapter_type": "parallel", "downsample_factor": 4, "add_layernorm": True, **kwargs},
            #     "attn": {"adapter_type": "normal", "l": 20, **kwargs},
            # }
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

        # freeze appropriate params
        LAYERNORM_NAMES = ["ln_f", "ln_1", "ln_2"]
        if config.freeze_lm:
            for name, param in self.lm.named_parameters():  # freeze lm weights
                if config.adapter_config and "adapter" in name:
                    param.requires_grad = True
                elif not config.freeze_layernorms and any(
                    [n in name for n in LAYERNORM_NAMES]
                ):
                    param.requires_grad = True
                elif config.tune_lm_biases and name.endswith("bias"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if config.freeze_img_encoder:
            for param in self.image_prefix.enc.parameters():
                param.requires_grad = False
        if config.tune_img_encoder_biases:
            for name, param in self.image_prefix.enc.named_parameters():
                if name.endswith("bias"):
                    param.requires_grad = True
        if config.freeze_img_encoder_batchnorms:
            for name, param in self.image_prefix.enc.named_parameters():
                if re.search("bn[\d+]", name):
                    param.requires_grad = False

    def add_adapters(
        self,
        downsample_factor: int = 4,
        transformer_attr: str = "transformer",
        ff_attr: str = "mlp",
        attn_attr: str = "attn",
        adapter_type: Literal["normal", "parallel", "scaled_parallel"] = "normal",
        location: Literal["mlp", "attention"] = "mlp",
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

        layers = getattr(self, transformer_attr)
        n_layers = len(layers)

        for l in range(n_layers):
            if location == "mlp":
                if self.mlp_adapter_added:
                    raise ValueError("Adapter layer already added")
                mlp = getattr(layers[l], ff_attr)
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
                setattr(layers[l], ff_attr, adapter_layer)
            else:
                if self.attn_adapter_added:
                    raise ValueError("Adapter layer already added")
                attn = getattr(layers[l], attn_attr)
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
                setattr(layers[l], attn_attr, adapter_layer)

        if location == "mlp":
            self.mlp_adapter_added = True
        else:
            self.attn_adapter_added = True

    def embed(self, imgs_words: List[torch.Tensor]) -> TensorType["b", "s", "d"]:
        """
        Embeds a list of tensors In the correct format to input into the LM (b, s, d).
        For each tensor, if it's 2d assume it's text and use word embedding,
        if it's 4d, assume it's an image, and use image_prefix to embed.
        """
        emb_list = []
        for x in imgs_words:
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
        filter_logits_fn: Callable = top_k,
        filter_threshold: float = 0.9,
        eos_token: int = None,
        decode: bool = True,
        remove_tokens_after_eos: bool = True,
    ):
        """
        Generates captions for a batch of embeddings.
        """

        # init values
        eos_token = eos_token or self.eos_token
        was_training = self.training
        self.eval()
        b, s, d = embeddings.shape
        past_key_values = None

        # init output with image tokens
        out = torch.zeros((b, s), dtype=torch.int64).to(self.device) + self.image_token

        # do sampling
        for i in range(max_steps):
            if i == 0:
                outputs = self.lm(
                    inputs_embeds=embeddings,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
            else:
                x = out[:, -1:]
                outputs = self.lm(
                    input_ids=x, use_cache=True, past_key_values=past_key_values
                )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            # filter / temperature sample
            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_threshold)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            try:
                sample = torch.multinomial(probs, 1)
            except RuntimeError:
                # nan in probs
                break

            out = torch.cat((out, sample), dim=-1)

            if eos_token is not None and (sample == eos_token).all():
                break

        if decode:
            captions = []
            for b in out:
                if remove_tokens_after_eos:
                    # any tokens after and end of sequence token is produced are also set to the eos token
                    eos_index = (b == eos_token).nonzero()
                    if eos_index.any():
                        b[eos_index[0] :] = eos_token
                b = b.tolist()
                b = [
                    i for i in b if (not i == self.image_token) and (not i == eos_token)
                ]
                caption = self.tokenizer.decode(b)
                captions.append(caption)
            out = captions

        self.train(was_training)
        return out

    def build_labels(
        self,
        input_embeddings: TensorType["b", "s", "d"],
        captions: TensorType["b", "s"],
    ) -> TensorType["b", "s"]:
        """
        Builds labels from input embeddings.
        Masks out the labels with -100 in positions up to the seq length of the embeddings, so only captions are predicted.
        Additionally, masks out everything *after* the first eos token.
        """
        shape = input_embeddings.shape[:2]  # b, s

        assert captions.shape[1] >= shape[1]

        # make sure to add masked embedding tokens in the appropriate locations in the labels
        embedding_tokens = torch.zeros(shape, dtype=torch.int64).to(self.device) - 100
        labels = torch.cat(
            (embedding_tokens, captions[:, : -shape[1]]), dim=1
        )  # we truncate the sequence length of the captions, as they are always padded to the full sequence length

        # mask out repeating eos tokens
        for label in labels:
            for k, token in enumerate(label):
                if token == self.eos_token:
                    label[k + 1 :] = -100
                    break

        return labels

    def forward(
        self,
        images: TensorType["b", "c", "h", "w"] = None,
        captions: Optional[TensorType["b", "seq"]] = None,
        inference: bool = False,
        output_hidden_states: bool = False,
        input_embeddings: TensorType["b", "s", "d"] = None,
    ) -> ModelOutput:
        if not inference:
            assert captions is not None, "Must provide captions during training"
        assert any([i is not None for i in [images, input_embeddings]]) and not all(
            [i is not None for i in [images, input_embeddings]]
        ), "Pass in either images, or input embeddings, not both."
        if input_embeddings is None:
            input_embeddings = self.image_prefix(images)
        if inference:
            return self.generate(input_embeddings)
        else:
            assert (
                captions.shape[1] == self.seq_len
            ), f"in training, captions should be padded to sequence length ({self.seq_len}), but are length {captions.shape[1]}"

            attn_adapter_config = self.config.adapter_config.get("attention", {})

            labels = self.build_labels(
                input_embeddings, captions
            )  # build labels from images or input_embeddings
            word_embeddings = self.word_embedding(captions)

            # join together
            input_embeddings = torch.cat(
                (
                    input_embeddings,
                    word_embeddings[:, : -input_embeddings.shape[1], :],
                ),  # remove padding in the word embedding before concatenating
                dim=1,
            )
            # feed joined embeddings through pretrained lm
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

    model = MultimodalLM(
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


# ------------------------- Classification wrapper class ----------------------------------


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes, classifier_type):
        super().__init__()

        if classifier_type == "linear":
            self.module = nn.Linear(hidden_size, num_classes)
        else:
            raise ValueError(f"invalid argument classifier_type = {classifier_type}")

    def forward(self, x):
        return self.module(x)


class MultimodalClassifier(MultimodalLM):
    def __init__(
        self, lm: nn.Module, tokenizer, config, device=None,
    ):
        super().__init__(lm, tokenizer, config, device)

        self.class_dict = self.config.class_dict
        self.num_classes = self.class_dict["num_classes"]
        self.classifier_type = self.class_dict.get("classifier_type", "linear")
        self.interface_type = self.class_dict.get("interface_type", "last_hidden_state")
        self.interface_position = self.class_dict.get("interface_position", -1)
        if self.class_dict.get("freeze_model", False):
            # TODO turn off dropout for the model if it is frozen?
            for p in self.model.parameters():
                p.requires_grad = False

        self.class_head = ClassificationHead(
            self.lm.config.hidden_size, self.num_classes, self.classifier_type
        )

    # x = captions, shape=[b, s]
    def build_weight_mask(self, x, num_imgs=1):
        """
        Builds a weight mask from text input x [b,s] => w [b, s, d] that gives the average or last non eos
        embedding after contraction with the hidden states.
        """

        w = (x != self.tokenizer.eos_token_id).long()

        # padding corresponding to the prefix length
        w_prefix = torch.zeros(
            x.shape[0],
            self.image_prefix_seq_len * num_imgs,
            device=self.device,
            dtype=torch.long,
        )

        if self.interface_type == "average_hidden_state":
            w = w / torch.sum(w, dim=1).unsqueeze(dim=1)
            w = w.to(device=self.device, dtype=torch.float16)

        elif self.interface_type == "last_hidden_state":
            w_shifted = torch.cat(
                [w[:, 1:], torch.zeros(x.shape[0], 1, device=self.device)], dim=1
            )
            w = (w - w_shifted).to(device=self.device, dtype=torch.float16)
            # if the mask technique works properly, we should only see one nonzero value in each batch of w. If not, something is wrong.
            assert all(w.count_nonzero(dim=1) == 1), "Something has gone wrong"

        # TODO: What to do about the case where in the last_hidden_state computation the index is larger than seq_len - img_seq_len so it gets pushed out in the next line?
        # append
        # prepend prefix padding
        w = torch.cat([w_prefix, w[:, : -self.image_prefix_seq_len * num_imgs]], dim=1)
        if all(w.sum(dim=1) == 0):
            print("Warning: Input length exceeded maximum sequence length")

        return w

    @classmethod
    def from_pretrained(
        cls,
        config,
        model_dir="/mnt/localdisk/models",
        tokenizer_name="gpt2",
        device=None,
    ):
        tokenizer = get_tokenizer(tokenizer_name)
        model = cls(
            get_language_model(
                config.lm_name, model_dir=model_dir, from_pretrained=False, no_init=True
            ),
            tokenizer,
            config,
            device,
        )

        ckpt_path = config.class_dict.get("pretrained_checkpoint")
        load_strict = False

        classification_ckpt_path = config.load
        if classification_ckpt_path is not None:
            # load latest checkpoint if one exists
            ckpt_path = infer_checkpoint_path_from_config(config)
            load_strict = True

        assert ckpt_path is not None
        print(f"loading multimodal transformer checkpoint...")
        state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))["module"]
        model.load_state_dict(state_dict, strict=load_strict)
        print(f"loaded multimodal transformer from checkpoint {ckpt_path}")

        return model

    def forward(self, images, captions, labels, return_probs=True):

        # images = [l_images, r_images] for nlvr2
        if not isinstance(images, list):
            images = [images]

        embeddings = self.embed(
            images + [captions[:, : -self.image_prefix_seq_len * len(images)]]
        )

        # embeddings = self.embed([images, captions[:, : -self.image_prefix_seq_len]])

        lm_out = self.lm(inputs_embeds=embeddings, output_hidden_states=True)

        hidden_states = lm_out.hidden_states[self.interface_position]

        w = self.build_weight_mask(captions, num_imgs=len(images))

        class_embeddings = torch.einsum("bsd, bs -> bd", hidden_states, w)

        logits = self.class_head(class_embeddings)

        loss = F.cross_entropy(logits, labels)

        if return_probs:
            return loss, F.softmax(logits, dim=1)

        return loss
