import torch
from transformers import (
    GPTNeoForCausalLM,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPT2Config,
    GPTJConfig,
    GPTNeoConfig,
)
from .utils import print_main
from transformers.modeling_utils import no_init_weights
import contextlib

LANGUAGE_MODELS = [
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-125M",
    "gpt2",
    "EleutherAI/gpt-j-6B",
]


def _get_hf_config_class(name):
    if "gpt2" in name:
        return GPT2Config

    elif "gpt-j" in name:
        return GPTJConfig

    else:
        return GPTNeoConfig


def _get_hf_model_class(name):
    if "gpt2" in name:
        return GPT2LMHeadModel

    elif "gpt-j" in name:
        return GPTJForCausalLM

    else:
        return GPTNeoForCausalLM


def _load_hf_model(config, name, from_pretrained=False, cache_dir=None):
    model_class = _get_hf_model_class(name)

    if from_pretrained:
        print_main("loading language model from pretrained weights...")
        return model_class.from_pretrained(config, cache_dir=cache_dir)
    else:
        print_main("initializing language model without pretrained weights...")
        return model_class(config)


def get_language_model(
    name: str,
    gradient_checkpointing: bool = True,
    training=True,
    from_pretrained=True,
    model_dir="/data/models/",
    no_init=True,
) -> torch.nn.Module:
    """
    Loads language model from HF
    """

    assert name in LANGUAGE_MODELS, f"{name} is not a valid language model"

    # config = AutoConfig.from_pretrained(name, cache_dir=model_dir)
    config = _get_hf_config_class(name).from_pretrained(name)
    config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        config.use_cache = False
    config.model_device = "cpu"

    ctx = no_init_weights if no_init else contextlib.nullcontext
    with ctx():
        # if "gpt2" in name:
        #     model = GPT2LMHeadModel.from_pretrained(
        #         None if not from_pretrained else name,
        #         config=config,
        #         cache_dir=model_dir,
        #     )
        # elif "gpt-j" in name:
        #     model = GPTJForCausalLM.from_pretrained(
        #         name,
        #         config=config,
        #         cache_dir=model_dir,
        #     )
        # else:
        #     model = GPTNeoForCausalLM.from_pretrained(
        #         None if not from_pretrained else name,
        #         config=config,
        #         cache_dir=model_dir,
        #     )
        model = _load_hf_model(
            config, name, from_pretrained=from_pretrained, cache_dir=model_dir
        )

    print_main("done!")

    if training:
        model.train()
    return model
