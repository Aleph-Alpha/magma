from dataclasses import dataclass, asdict
import yaml
from pprint import pprint
from .utils import is_main
import os
from pathlib import Path

def load_config(path, config_dir=Path('configs')):
    if not path.endswith('.yml'):
        path += '.yml'
    if not os.path.exists(path):
        path = config_dir / path
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


@dataclass
class MultimodalConfig:

    batch_size: int
    train_steps: int
    name: str = None
    optimizer_name: str = "AdamW"
    encoder_name: str = "clip"
    tokenizer_name: str = "gpt2"
    lm_name: str = "gptj"
    train_dataset_name: str = "conceptual_captions"
    eval_dataset_name: str = "/data/conceptual_captions"
    train_dataset_dir: str = "/data/coco_data"
    eval_dataset_dir: str = "/data/coco_data"
    vqa_dir: str = "/data/vqa"
    gqa_dir: str = "/data/gqa"
    proj_layer_type: str = "linear"
    freeze_lm: bool = True
    freeze_img_encoder: bool = True
    image_size: int = 256
    image_seq_len: int = 2
    save: str = None
    load: str = None
    lr: float = 8.0e-4
    image_enc_lr: float = None
    min_lr: float = 0.0
    lr_decay_iters: int = None
    log_every: int = 1
    eval_every: int = 250
    save_every: int = 2500
    gradient_accumulation_steps: int = 1
    train_size: int = None
    eval_size: int = None
    eval_steps: int = 25
    extra_augmentations: bool = False
    zero_stage: int = 2
    gradient_clipping: float = 1.0
    warmup_num_steps: int = 100
    pretrained_img_encoder: bool = False
    adaptive_gradient_clipping: float = 0.0
    freeze_layernorms: bool = True
    image_embed_dropout_prob: float = 0.0
    use_image_embed_layernorm: bool = False
    adapter_config: dict = None
    adapter_downsample_factor: int = 4
    weight_decay: float = 0.00
    tune_lm_biases: bool = False
    tune_img_encoder_biases: bool = False
    freeze_img_encoder_batchnorms: bool = True
    seq_len: int = None
    run_blind: bool = False
    wandb_project: str = "multimodal_transformer"
    fine_tune: bool = False
    dataset_type: str = 'old'
    load_optimizer: bool = True

    #configs for classification class. 
    class_dict: dict = None #{num_classes: .., ckpt_path: .., classifier_type:, .., interface_type: .., interface_position: .., freeze_model: ..}

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
                }}
        else:
            self.lr_scheduler = "WarmupDecayLR"
            self.scheduler_dict = {
                "type": self.lr_scheduler,
                "params": {
                    "total_num_steps": self.lr_decay_iters,
                    "warmup_min_lr": self.min_lr,
                    "warmup_max_lr": self.lr,
                    "warmup_num_steps": self.warmup_num_steps,
                }}
        self.deepspeed_config_params = {
            "train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.gradient_clipping,
            "fp16": {"enabled": True, "loss_scale_window": 250},
            "scheduler": self.scheduler_dict,
            "zero_optimization": {"stage": self.zero_stage, "load_from_fp32_weights": False},
        }
        assert self.gradient_clipping == 0.0 or self.adaptive_gradient_clipping == 0.0
        if self.name is None:
            # set automatic name
            enc_str = f"encoder-{self.encoder_name}"
            if self.pretrained_img_encoder:
                enc_str += "-pretrained"
            name_params = [enc_str, f"lr-{self.lr:.2e}", f"bs-{self.batch_size}"]
            non_defaults = []
            if not self.is_default('freeze_lm'):
                non_defaults.append('tune-lm')
            if not self.is_default('freeze_img_encoder'):
                non_defaults.append('tune-img-encoder')
            if not self.is_default('adapter_config'):
                non_defaults.append(str(self.adapter_config))
            if not self.is_default('image_enc_lr'):
                non_defaults.append(f'img-enc-lr-{self.image_enc_lr}')
            if self.lr_decay_iters is not None:
                non_defaults.append(f'lr-decay-{self.lr_decay_iters}-steps')
            if not self.is_default('image_embed_dropout_prob'):
                non_defaults.append(f'img-embed-dropout-{self.image_embed_dropout_prob}')
            if not self.is_default('use_image_embed_layernorm'):
                non_defaults.append(f'img-embed-ln')
            if not self.is_default('freeze_layernorms'):
                non_defaults.append('tune_layernorms')
            if not self.is_default('image_seq_len'):
                non_defaults.append(f'image-seq-{self.image_seq_len}')
            name_params += non_defaults
            self.name = "_".join(name_params)

    @classmethod
    def from_yml(cls, path):
        return cls(**load_config(path))

    def to_dict(self):
        return asdict(self)


@dataclass
class MMHSConfig:

    batch_size: int
    train_steps: int
    name: str = None
    question_prompt: str = "Q: Is this meme hateful? A:"
    use_text: bool = True
    optimizer_name: str = "AdamW"
    classifier_type: str = 'mlp'
    data_dir: str = '/data'
    train_datasets: tuple = 'train_mmhs', 'train_hm'
    eval_datasets: tuple = 'val_mmhs', 'test_mmhs', 'dev_hm'
    split: int = None
    # train_dataset_name: str = "mmhs"
    # train_dataset_dir: str = "/data/MMHS"
    # train_dataset_split: str = "train"
    # eval_dataset_name: str = "hm"
    # eval_dataset_dir: str = "/data/hateful_memes"
    # eval_dataset_split: str = "train"
    save: str = None
    load: str = None
    mm_config_dir: str = None
    freeze_mm: bool = True
    load_mm: str = None
    # lm_interface_type: str = '2tolast_layer_last_hidden_state'
    lr: float = 1e-4
    min_lr: float = 0.0
    lr_decay_iters: int = None
    use_mixed_precision: bool = False
    log_every: int = 100
    eval_every: int = 5000
    save_every: int = 1000
    gradient_accumulation_steps: int = 1
    # train_size: int = None
    # eval_size: int = None
    eval_steps: int = None
    zero_stage: int = 2
    gradient_clipping: float = 1.0
    warmup_num_steps: int = 1000
    adaptive_gradient_clipping: float = 0.0

    def print(self):
        if is_main():
            print("-" * 100)
            pprint(self.__dict__, indent=4)
            print("-" * 100)

    def __post_init__(self):
        if self.lr_decay_iters is None:
            self.lr_decay_iters = self.train_steps
        assert self.gradient_clipping == 0.0 or self.adaptive_gradient_clipping == 0.0

    @classmethod
    def from_yml(cls, path):
        return cls(**load_config(path))

    def to_dict(self):
        return asdict(self)
