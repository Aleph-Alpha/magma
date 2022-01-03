import os
import torch
from transformers import GPTNeoForCausalLM, AutoConfig, GPT2LMHeadModel
import gdown
import tarfile
from .utils import is_main, Checkpoint, print_main
from pathlib import Path
from transformers.modeling_utils import no_init_weights

LANGUAGE_MODELS = [
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-125M",
    "gpt2",
    "gptj",
]


def gptj_config():
    config = AutoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
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


def download_gptj(
    model_dir: str = "/data/models/",
    url: str = "https://drive.google.com/uc?id=1VXXCMR_ETxOd3rxG4eXxS4-QA5NekB3H",  # backup url: s3://aleph-alpha34rtgyhu/neo_6b.tar
):
    os.makedirs(model_dir, exist_ok=True)
    ckpt_dir = Path(model_dir) / "j6b_ckpt"
    output_file = Path(model_dir) / "neo_6b.tar"
    if not os.path.isfile(output_file) or not os.path.isdir(ckpt_dir):
        if is_main():
            if not os.path.isfile(output_file):
                if "drive.google" in url:
                    gdown.download(url, str(output_file), quiet=False)
                elif "s3://" in url:
                    os.system(f"aws s3 cp {url} {str(output_file)}")
                else:
                    os.system(f"wget {url} -P {model_dir}")
            with tarfile.open(name=output_file) as tar:
                print("Extracting tar...")
                tar.extractall(model_dir)
                print("Done!")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


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
    print_main("Loading language model from checkpoint...")
    assert (
        name in LANGUAGE_MODELS
    ), f"{name} not recognized - please choose from:\n\t{LANGUAGE_MODELS}"
    if name != "gptj":
        config = AutoConfig.from_pretrained(name)
    else:
        config = gptj_config()
    config.gradient_checkpointing = gradient_checkpointing
    if gradient_checkpointing:
        config.use_cache = False
    config.model_device = "cpu"
    if "gpt-neo" in name:
        model = GPTNeoForCausalLM.from_pretrained(name, config=config)
    elif name == "gptj":
        if from_pretrained:
            download_gptj(model_dir)
            ckpt_path = Path(model_dir) / "j6b_ckpt"
            model = GPTNeoForCausalLM.from_pretrained(
                pretrained_model_name_or_path=None,
                config=config,
                state_dict=Checkpoint(str(ckpt_path), device="cpu"),
            )
        else:
            if no_init:
                print("Loading GPT model with no initialization")
                with no_init_weights():
                    model = GPTNeoForCausalLM(config=config)
            else:
                print("Initializing model with random weights")
                model = GPTNeoForCausalLM(config=config)
    else:
        model = GPT2LMHeadModel.from_pretrained(name, config=config)
    print_main("Done loading model!")
    if training:
        model.train()
    return model