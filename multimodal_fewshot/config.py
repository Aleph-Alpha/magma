from dataclasses import dataclass, asdict
import yaml
from pprint import pprint
from .utils import is_main
import os
from pathlib import Path


def load_config(path, config_dir=Path("configs")):
    if not path.endswith(".yml"):
        path += ".yml"
    if not os.path.exists(path):
        path = config_dir / path
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


@dataclass
class MultimodalConfig:

    # Training:
    # ------------------------------------------------------------

    batch_size: int
    train_steps: int
    optimizer_name: str = "AdamW"
    lr: float = 8.0e-4
    image_enc_lr: float = None
    min_lr: float = 0.0
    lr_decay_iters: int = None
    gradient_accumulation_steps: int = 1
    image_size: int = 256
    eval_every: int = 250
    eval_steps: int = 25
    zero_stage: int = 2
    gradient_clipping: float = 1.0
    warmup_num_steps: int = 100
    weight_decay: float = 0.00
    run_blind: bool = False
    fine_tune: bool = False
    load_optimizer: bool = True

    # Checkpointing:
    # ------------------------------------------------------------
    save_every: int = 2500
    save: str = None
    load: str = None

    # Data:
    # ------------------------------------------------------------
    train_dataset_name: str = "conceptual_captions"
    eval_dataset_name: str = "/data/conceptual_captions"
    train_dataset_dir: str = "/data/coco_data"
    eval_dataset_dir: str = "/data/coco_data"
    vqa_dir: str = "/data/vqa"
    gqa_dir: str = "/data/gqa"

    # Model architecture:
    # ------------------------------------------------------------
    encoder_name: str = "clip"
    tokenizer_name: str = "gpt2"
    lm_name: str = "gptj"
    image_seq_len: int = 2
    pretrained_img_encoder: bool = False
    seq_len: int = None

    # Layer Freezing settings:
    # ------------------------------------------------------------
    freeze_lm: bool = True
    freeze_img_encoder: bool = True
    freeze_layernorms: bool = True
    image_embed_dropout_prob: float = 0.0
    use_image_embed_layernorm: bool = False
    tune_lm_biases: bool = False
    tune_img_encoder_biases: bool = False
    freeze_img_encoder_batchnorms: bool = True

    # Adapter settings:
    # ------------------------------------------------------------
    adapter_config: dict = None

    # Classification Finetuning settings:
    # ------------------------------------------------------------
    class_dict: dict = None  # {num_classes: .., ckpt_path: .., classifier_type:, .., interface_type: .., interface_position: .., freeze_model: ..}

    # Logging settings:
    # ------------------------------------------------------------
    name: str = None  # name, just used for wandb logging
    log_every: int = 1
    wandb_project: str = "magma"

    def print(self):
        if is_main():
            print("-" * 100)
            pprint(self.__dict__, indent=4)
            print("-" * 100)

    def is_default(self, arg: str):
        return getattr(self, arg) == self.__dataclass_fields__.get(arg).default

    def __post_init__(self):
        self.is_classifier = self.class_dict is not None
        if self.adapter_config is None:
            self.adapter_config = {}
        if self.lr_decay_iters is None:
            self.lr_scheduler = "WarmupLR"
            self.scheduler_dict = {
                "type": self.lr_scheduler,
                "params": {
                    "warmup_min_lr": self.min_lr,
                    "warmup_max_lr": self.lr,
                    "warmup_num_steps": self.warmup_num_steps,
                },
            }
        else:
            self.lr_scheduler = "WarmupDecayLR"
            self.scheduler_dict = {
                "type": self.lr_scheduler,
                "params": {
                    "total_num_steps": self.lr_decay_iters,
                    "warmup_min_lr": self.min_lr,
                    "warmup_max_lr": self.lr,
                    "warmup_num_steps": self.warmup_num_steps,
                },
            }
        self.deepspeed_config_params = {
            "train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.gradient_clipping,
            "fp16": {"enabled": True, "loss_scale_window": 250},
            "scheduler": self.scheduler_dict,
            "zero_optimization": {
                "stage": self.zero_stage,
                "load_from_fp32_weights": False,
            },
        }
        if self.name is None:
            # derive automatic name for wandb logging
            enc_str = f"encoder-{self.encoder_name}"
            if self.pretrained_img_encoder:
                enc_str += "-pretrained"
            name_params = [enc_str, f"lr-{self.lr:.2e}", f"bs-{self.batch_size}"]
            non_defaults = []
            if not self.is_default("freeze_lm"):
                non_defaults.append("tune-lm")
            if not self.is_default("freeze_img_encoder"):
                non_defaults.append("tune-img-encoder")
            if not self.is_default("adapter_config"):
                non_defaults.append(str(self.adapter_config))
            if not self.is_default("image_enc_lr"):
                non_defaults.append(f"img-enc-lr-{self.image_enc_lr}")
            if self.lr_decay_iters is not None:
                non_defaults.append(f"lr-decay-{self.lr_decay_iters}-steps")
            if not self.is_default("image_embed_dropout_prob"):
                non_defaults.append(
                    f"img-embed-dropout-{self.image_embed_dropout_prob}"
                )
            if not self.is_default("use_image_embed_layernorm"):
                non_defaults.append(f"img-embed-ln")
            if not self.is_default("freeze_layernorms"):
                non_defaults.append("tune_layernorms")
            if not self.is_default("image_seq_len"):
                non_defaults.append(f"image-seq-{self.image_seq_len}")
            name_params += non_defaults
            self.name = "_".join(name_params)

    @classmethod
    def from_yml(cls, path):
        return cls(**load_config(path))

    def to_dict(self):
        return asdict(self)
