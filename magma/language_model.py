import torch
from transformers import GPTNeoForCausalLM, AutoConfig, GPT2LMHeadModel, AutoModelForCausalLM
from .utils import print_main
from pathlib import Path

from transformers.modeling_utils import no_init_weights

LANGUAGE_MODELS = [
    "gptj",
]


def gptj_config():
    config = AutoConfig.from_pretrained(
        "/home/ml-mmeuer/adaptable_magma/model_checkpoints/gpt-neo-1.3B", local_files_only=True)
    config.attention_layers = ["global"] * 28
    config.attention_types = [["global"], 28]
    config.num_layers = 28
    config.num_heads = 16
    config.hidden_size = 256 * config.num_heads
    config.vocab_size = 50400
    config.rotary = True
    config.rotary_dim = 64
    config.jax = True
    config.gradient_checkpointing = True
    return config


def get_gptj(
    gradient_checkpointing: bool = True,
    from_pretrained=False,
) -> torch.nn.Module:
    """
    Loads GPTJ language model from HF
    """
    config = gptj_config()
    config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        config.use_cache = False
    config.model_device = "cpu"

    with no_init_weights():
        # with init_empty_weights():
        print_main("Fetching GPTJ language model...")
        model = GPTNeoForCausalLM(config=config)
    print("Done Fetching Model ... ")
    return model
