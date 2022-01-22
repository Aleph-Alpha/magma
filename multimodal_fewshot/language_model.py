import torch
from transformers import GPTNeoForCausalLM, GPT2LMHeadModel, GPTJForCausalLM, AutoConfig
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

    print_main("Loading language model from checkpoint...")
    config = AutoConfig.from_pretrained(name, cache_dir=model_dir)
    config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        config.use_cache = False
    config.model_device = "cpu"

    ctx = no_init_weights if no_init else contextlib.nullcontext
    with ctx():
        if "gpt2" in name:
            model = GPT2LMHeadModel.from_pretrained(
                None if not from_pretrained else name,
                config=config,
                cache_dir=model_dir,
            )
        elif "gpt-j" in name:
            model = GPTJForCausalLM.from_pretrained(
                None if not from_pretrained else name,
                config=config,
                cache_dir=model_dir,
            )
        else:
            model = GPTNeoForCausalLM.from_pretrained(
                None if not from_pretrained else name,
                config=config,
                cache_dir=model_dir,
            )

    if training:
        model.train()
    return model
