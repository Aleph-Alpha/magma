# interface to run eval on all tasks:

# takes in a model, task name, and whether to use few shot
# args will be:
#   config
#   (optional) - checkpoint path. If no path is provided, look in the 'save' directory of the config.
#   (optional) - task name. If no task name is provided, use all suitable tasks.
#   (optional) - task config. Provides details on how to run the tasks. Number of few shots, temperature etc.
#   (optional) - save_path. Provides a path to save the results. If no path is provided, just print to stdout.

# Task interface:
#   - maintain a task registry, which contains all the tasks and their details.
#   - Each task should have a tasktype - 'classification' / 'generative'
#   - Each task should have a boolean 'few_shot' - whether it is a few shot task or not.
#   - Each task should have a unified interface: a function that takes in a model, a task config, and a few shot config,
#     and returns a dictionary of results.


import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal, List
from pprint import pprint
import json
from multimodal_fewshot.datasets.dataset import ImgCptDataset
from tqdm import tqdm
from functools import partial
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from multimodal_fewshot.model import get_multimodal_model, MultimodalClassifier
from multimodal_fewshot.config import MultimodalConfig
from multimodal_fewshot.datasets.vqa_eval import run_vqa_eval
from multimodal_fewshot.datasets.coco import eval_coco
from multimodal_fewshot.datasets.nocaps import nocaps_eval
from multimodal_fewshot.utils import (
    get_world_info,
    init_distributed,
    get_tokenizer,
    reduce_losses,
    collate_fn_classification,
    infer_checkpoint_path_from_config,
    to_cuda_half,
)
from multimodal_fewshot.transforms import get_transforms
import torch

from torch.utils.data import Subset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import random
from multimodal_fewshot.datasets.snli_ve import SNLI_VE_Dataset
from multimodal_fewshot.datasets.nlvr2 import NLVR2Dataset
from multimodal_fewshot.datasets import ImgCptDataset
from multimodal_fewshot.datasets.dataset import ClassificationWrapper


def get_classification_dataset_class(dataset_name):
    if dataset_name == "snli_ve":
        return SNLI_VE_Dataset
    elif dataset_name == "nlvr2":
        return NLVR2Dataset
    else:
        raise ValueError(f"dataset class {dataset_name} not recognized")


def infer_dataset_path(data_dir, dataset_name):

    path = Path(data_dir) / dataset_name
    if path.exists():
        return str(path)

    raise ValueError(
        f"Could not find dataset at {data_dir}. Please provide dataset path for the dataset explicitly"
    )


def load_classification_dataset(
    root_data_dir,
    dataset_name,
    transforms,
    tokenizer,
    world_size,
    rank,
    mode=None,
    max_n_steps=None,
    batch_size=16,
    collate_fn=None,
    num_classes=None,
):

    dataset_class = get_classification_dataset_class(dataset_name)
    dataset_path = infer_dataset_path(root_data_dir, dataset_name)

    if dataset_name == "snli_ve":
        assert mode is not None
        dataset = dataset_class(
            dataset_path, tokenizer=tokenizer, transforms=transforms, mode=mode
        )
    elif dataset_name == "nlvr2":
        assert mode is not None
        dataset = NLVR2Dataset(
            dataset_path, mode, tokenizer=tokenizer, transforms=transforms
        )
    else:
        assert num_classes is not None
        dataset = ImgCptDataset(
            dataset_path, tokenizer=tokenizer, transforms=transforms
        )
        dataset = ClassificationWrapper(dataset, num_classes)

    if max_n_steps is not None and len(dataset) > max_n_steps:
        dataset = Subset(dataset, random.sample(range(len(dataset)), max_n_steps))

    sampler = (
        DistributedSampler(dataset, shuffle=True, num_replicas=world_size, rank=rank)
        if world_size > 1
        else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        collate_fn=collate_fn,
    )

    return dataloader


def _classification_eval(model, dataset, dataset_name):

    accuracies = []

    for (images, caption, class_label) in tqdm(
        dataset, f"Running eval on {dataset_name}"
    ):
        images, caption, class_label = to_cuda_half(images, caption, class_label)
        # print(len(images))
        # print(images[0].shape)
        # print(caption.shape)
        # print(class_label.shape)
        loss, logits = model(images, caption, class_label)
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == class_label).float().mean()
        accuracies.append(accuracy)

    accuracy_reduced = reduce_losses(torch.mean(torch.stack(accuracies))).item()

    return {"accuracy": accuracy_reduced}


def classification_eval(
    dataset_name,
    model,
    tokenizer,
    transforms,
    data_dir,
    mode=None,
    max_n_steps=None,
):

    num_classes = None

    if dataset_name == "foil_val_converted":
        num_classes = 2

    local_rank, rank, world_size = get_world_info()
    dataset = load_classification_dataset(
        data_dir,
        dataset_name,
        transforms=transforms,
        tokenizer=tokenizer,
        mode=mode,
        world_size=world_size,
        rank=rank,
        max_n_steps=max_n_steps,
        collate_fn=collate_fn_classification,
        num_classes=num_classes,
    )
    return _classification_eval(
        model,
        dataset,
        dataset_name,
    )


TASK_REGISTRY = {
    "vqa": {
        "type": "generative",
        "function": partial(run_vqa_eval, vqa_dataset_name="vqa"),
        "is_few_shot": True,
    },
    "gqa": {
        "type": "generative",
        "function": partial(run_vqa_eval, vqa_dataset_name="gqa", mode="testdev"),
        "is_few_shot": True,
    },
    "okvqa": {
        "type": "generative",
        "function": partial(run_vqa_eval, vqa_dataset_name="okvqa"),
        "is_few_shot": True,
    },
    "vizwiz": {
        "type": "generative",
        "function": partial(run_vqa_eval, vqa_dataset_name="vizwiz"),
        "is_few_shot": True,
    },
    "openended": {
        "type": "generative",
        "function": partial(run_vqa_eval, vqa_dataset_name="mini_image_net"),
        "is_few_shot": True,
    },
    "fast_vqa": {
        "type": "generative",
        "function": partial(run_vqa_eval, vqa_dataset_name="fast_vqa"),
        "is_few_shot": True,
    },
    "coco": {"type": "generative", "function": eval_coco, "is_few_shot": False},
    "nocaps": {"type": "generative", "function": nocaps_eval, "is_few_shot": False},
    "snli_ve": {
        "type": "classification",
        "function": partial(classification_eval, dataset_name="snli_ve", mode="test"),
        "is_few_shot": False,
    },
    "foil_val_converted": {
        "type": "classification",
        "function": partial(classification_eval, dataset_name="foil_val_converted"),
        "is_few_shot": False,
    },
    "nlvr2": {
        "type": "classification",
        "function": partial(
            classification_eval, dataset_name="nlvr2", mode="val"
        ),  # might want to use another split for eval
        "is_few_shot": False,
    },
}


@dataclass
class EvalConfig:

    task_type: Literal["classification", "generative"] = "classification"
    few_shot_examples: List[int] = None
    question_prompt: str = "Q: "
    answer_prompt: str = "A: "
    separator: str = " "
    task_induction: str = None
    repeats: int = 0
    temperature: float = 0.01
    logits_filter_fn: Literal["top_k", "top_p", None] = "top_k"
    max_n_steps: int = None

    def __post_init__(self):
        if self.few_shot_examples is None:
            self.few_shot_examples = [1, 3, 4, 8]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_args(cls, args):
        kwargs = {k: v for k, v in vars(args).items() if k in cls.__dataclass_fields__}
        return cls(**kwargs)

    def print(self):
        print("-" * 100)
        pprint(self.__dict__, indent=4)
        print("-" * 100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument(
        "--checkpoint", type=str, help="path to checkpoint file", default=None
    )
    parser.add_argument(
        "--data_dir", type=str, help="path to data directory", default="/mnt/localdisk/"
    )
    parser.add_argument(
        "--tasks", type=str, help="task name(s)", default=None, nargs="+"
    )
    parser.add_argument(
        "--save_path", type=str, help="path to save results", default=None
    )
    parser.add_argument(
        "--task_type",
        type=str,
        help="type of tasks",
        choices=["classification", "generative"],
        default="generative",
    )
    parser.add_argument(
        "--few_shot_examples",
        type=int,
        help="number of examples per task. Pass multiple values to run the same model / task at different few shot settings",
        default=[0],
        nargs="+",
    )
    parser.add_argument(
        "--question_prompt",
        type=str,
        help="question prompt",
        default="Q: ",
    )
    parser.add_argument(
        "--answer_prompt", type=str, help="answer prompt", default="A: "
    )
    parser.add_argument(
        "--separator",
        type=str,
        help="separator between question and answer",
        default=" ",
    )
    parser.add_argument(
        "--task_induction", type=str, help="task induction", default=None
    )
    parser.add_argument("--repeats", type=int, help="number of repeats", default=0)
    parser.add_argument("--temperature", type=float, help="temperature", default=0.01)
    parser.add_argument(
        "--logits_filter_fn",
        type=str,
        help="logits filter function",
        choices=["top_k", "top_p", None],
        default="top_k",
    )
    parser.add_argument(
        "--max_n_steps", type=int, help="max number of steps", default=None
    )
    parser.add_argument(
        "--local_rank", type=int, help="local rank for multigpu processing", default=-1
    )
    return parser.parse_args()


@torch.no_grad()
def run_eval(
    model,
    tokenizer,
    transforms,
    data_dir=None,
    tasks=None,
    few_shot_examples=0,
    question_prompt="Q: ",
    answer_prompt="A: ",
    separator=" ",
    task_induction=None,
    repeats=0,
    temperature=0.01,
    logits_filter_fn="top_k",
    max_n_steps=None,
    task_specific_kwargs=None,
):
    results = {}

    # init distributed
    init_distributed()

    for task in tasks:

        if few_shot_examples > 0 and not TASK_REGISTRY[task]["is_few_shot"]:
            print(f"{task} is not a few shot task, skipping")
            results[task] = {"message": "not a few shot task"}
            continue

        eval_fn = TASK_REGISTRY[task]["function"]

        if TASK_REGISTRY[task]["type"] == "generative":
            kwargs = {
                "model": model,
                "tokenizer": tokenizer,
                "transforms": transforms,
                "few_shot_examples": few_shot_examples,
                "question_prompt": question_prompt,
                "answer_prompt": answer_prompt,
                "separator": separator,
                "task_induction": task_induction,
                "repeats": repeats,
                "temperature": temperature,
                "logits_filter_fn": logits_filter_fn,
                "max_n_steps": max_n_steps,
                "data_dir": data_dir,
            }
            if task_specific_kwargs is not None:
                if task_specific_kwargs.get(task) is not None:
                    kwargs.update(task_specific_kwargs[task])
            results[task] = eval_fn(**kwargs)
        else:
            kwargs = {
                "model": model,
                "tokenizer": tokenizer,
                "transforms": transforms,
                "max_n_steps": max_n_steps,
                "data_dir": data_dir,
            }
            if task_specific_kwargs is not None:
                if task_specific_kwargs.get(task) is not None:
                    kwargs.update(task_specific_kwargs[task])
            results[task] = eval_fn(**kwargs)

    return results


class Evaluator:
    def __init__(
        self,
        root_data_dir,
        few_shot_config,
        config_path=None,
        task_type="generative",
        checkpoint_path=None,
        tasks=None,
        save_path=None,
        task_specific_kwargs=None,
        model=None,
        transforms=None,
        tokenizer=None,
    ):
        self.task_type = task_type
        self.root_data_dir = root_data_dir
        assert task_type in ["classification", "generative"]
        self.config_path = config_path
        self.config = self.load_config()
        self.checkpoint_path = checkpoint_path
        self.tasks = tasks
        self.save_path = save_path
        self.local_rank, self.rank, self.world_size = init_distributed()
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.model = model
        self.model, self.transforms, self.tokenizer = self.load_model()
        # self.task_specific_kwargs = {"coco": {"return_captions": True}}
        self.task_specific_kwargs = {}
        self.few_shot_config = few_shot_config

    def load_config(self):
        return MultimodalConfig.from_yml(self.config_path)

    def info(self):
        info_dict = self.few_shot_config.to_dict()
        info_dict["tasks"] = self.tasks
        info_dict["task_type"] = self.task_type
        info_dict["checkpoint_path"] = self.checkpoint_path
        info_dict["model_config"] = self.config.to_dict()
        if self.task_specific_kwargs:
            info_dict["task_specific_kwargs"] = self.task_specific_kwargs
        return info_dict

    def load_model(self):
        model = None
        transforms = None
        tokenizer = None

        print(f"task type: {self.task_type}")

        if self.model is None:
            if self.task_type == "generative":
                # no model provided, load from disk
                if (
                    self.checkpoint_path is None
                ):  # if no checkpoint is provided, try to infer from config
                    self.checkpoint_path = infer_checkpoint_path_from_config(
                        self.config
                    )

                # load model, transforms and tokenizer
                model, transforms, tokenizer = get_multimodal_model(
                    config_path=self.config_path, ckpt_path=self.checkpoint_path
                )
                transforms = (
                    self.transforms if self.transforms is not None else transforms
                )  # default to transforms passed in if provided
                tokenizer = (
                    self.tokenizer if self.tokenizer is not None else tokenizer
                )  # default to tokenizer passed in if provided
            elif self.task_type == "classification":
                model = MultimodalClassifier.from_pretrained(self.config)
                print("loaded model with classification head")
        if model is None:
            # model provided, use it
            model = self.model

        if transforms is None:
            # load transforms and tokenizer separately
            if self.transforms is None:
                transforms = get_transforms(self.config.image_size, model=model)
            else:
                transforms = self.transforms

        if tokenizer is None:
            if self.tokenizer is None:
                tokenizer = get_tokenizer("gpt2")  # TODO make this configurable
            else:
                tokenizer = self.tokenizer

        model = model.half().cuda().eval()
        return model, transforms, tokenizer

    def run(self):
        if self.tasks is None:
            tasks = TASK_REGISTRY.keys()
            # split tasks by task type
            tasks = [t for t in tasks if TASK_REGISTRY[t]["type"] == self.task_type]
        else:
            tasks = self.tasks
        if not tasks:
            return {}

        results = {}

        for n_shots in self.few_shot_config.few_shot_examples:
            results[f"few_shot_{n_shots}"] = run_eval(
                self.model,
                tokenizer=self.tokenizer,
                transforms=self.transforms,
                tasks=self.tasks,
                few_shot_examples=n_shots,
                question_prompt=self.few_shot_config.question_prompt,
                answer_prompt=self.few_shot_config.answer_prompt,
                separator=self.few_shot_config.separator,
                task_induction=self.few_shot_config.task_induction,
                repeats=self.few_shot_config.repeats,
                temperature=self.few_shot_config.temperature,
                logits_filter_fn=self.few_shot_config.logits_filter_fn,
                max_n_steps=self.few_shot_config.max_n_steps,
                task_specific_kwargs=self.task_specific_kwargs,
                data_dir=self.root_data_dir,
            )

            if self.local_rank == 0:
                pprint(results)

            # save results
            if self.save_path is not None:
                print(f"Saving results to {self.save_path}")
                Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
                results.update({"info": self.info()})
                with open(self.save_path, "w") as f:
                    json.dump(results, f, indent=4)
            else:
                print("Skipping saving results as no save_path was provided")

        return results


if __name__ == "__main__":
    args = parse_args()
    evaluator = Evaluator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        root_data_dir=args.data_dir,
        few_shot_config=EvalConfig.from_args(args),
        task_type=args.task_type,
        tasks=args.tasks,
        save_path=args.save_path,
    )
    evaluator.run()
