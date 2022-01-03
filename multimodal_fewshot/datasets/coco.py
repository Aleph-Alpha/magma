import os
from torch.utils.data.sampler import BatchSampler
import torchvision
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import time

import torch
import json
import random

try:
    from .metrics import compute_nlg_metrics
except ImportError:
    from metrics import compute_nlg_metrics

# append main folder to sys path so we can import from main package
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from multimodal_fewshot.utils import get_world_info, is_main


def coco_collate_fn(batch):
    image_tensors = []
    captions = []
    for instance in batch:
        image_tensors.append(instance[0])
        captions.append(instance[1])
    return torch.cat(image_tensors, dim=0), captions


def download_coco(data_dir=None):
    """Downloads coco if it doesn't exist already"""
    if data_dir is None:
        data_dir = "./coco_data"
    if not os.path.isdir(f"{data_dir}/val2017"):
        if not os.path.isfile(f"{data_dir}/val2017.zip"):
            os.system(
                f"wget -P {data_dir} http://images.cocodataset.org/zips/val2017.zip"
            )
        os.system(f"cd {data_dir}; unzip val2017.zip; cd ..")
    if not os.path.isdir(f"{data_dir}/train2017"):
        if not os.path.isfile(f"{data_dir}/train2017.zip"):
            os.system(
                f"wget -P {data_dir} http://images.cocodataset.org/zips/train2017.zip"
            )
        os.system(f"cd {data_dir}; unzip train2017.zip; cd ..")
    if not os.path.isdir(f"{data_dir}/annotations"):
        if not os.path.isfile(f"{data_dir}/annotations_trainval2017.zip"):
            os.system(
                f"wget -P {data_dir} http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            )
        os.system(f"cd {data_dir}; unzip annotations_trainval2017.zip; cd ..")


def coco_dataset(data_dir=None, mode="train", transforms=None):
    "downloads / returns coco dataset"
    if data_dir is None:
        data_dir = "./coco_data"
    download_coco(data_dir=data_dir)
    root = Path(data_dir) / f"{mode}2017"
    ann_file = Path(data_dir) / "annotations" / f"captions_{mode}2017.json"
    return torchvision.datasets.CocoCaptions(
        root=str(root),
        annFile=str(ann_file),
        transform=transforms,
    )


def infer_dataset_path(data_dir):
    candidate_names = ["coco_data", "coco"]

    for name in candidate_names:
        path = Path(data_dir) / name
        if path.exists():
            return str(path)

    raise ValueError(
        f"Could not find dataset at {data_dir}. Please provide dataset path for coco explicitly"
    )


def eval_coco(
    model,
    tokenizer,
    transforms,
    data_dir=None,
    dataset_path=None,
    few_shot_examples=0,
    separator=" ",
    task_induction=None,
    repeats=0,
    temperature=0.01,
    logits_filter_fn="top_k",  # TODO: make sure this gets passed to the eval function
    max_n_steps=None,
    mode="val",
    batch_size=16,
    return_captions=False,
    **kwargs,
):
    if dataset_path is None:
        dataset_path = infer_dataset_path(data_dir)
    local_rank, rank, world_size = get_world_info()
    assert (
        world_size <= 8
    ), "Coco eval only supports up to 8 workers and they must all share the same filesystem"

    # load dataset
    ds = coco_dataset(data_dir=dataset_path, mode=mode, transforms=transforms)
    if max_n_steps is not None and len(ds) > max_n_steps:
        ds = Subset(ds, random.sample(range(len(ds)), max_n_steps))

    sampler = (
        DistributedSampler(ds, shuffle=True, num_replicas=world_size, rank=rank)
        if world_size > 1
        else None
    )

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        collate_fn=coco_collate_fn,
    )

    if few_shot_examples > 0:
        raise NotImplementedError("Few shot examples for COCO not implemented yet")

    hypotheses = []
    references = []
    for (images, captions) in tqdm(
        iter(dataloader),
        desc=f"Running Eval for COCO {mode} set",
        disable=rank > 0,
        total=len(ds) // batch_size // world_size,
    ):
        images = images.cuda().half()

        longest_caption = 0
        for caption_batch in captions:
            for caption in caption_batch:
                if len(tokenizer.encode(caption, truncation=True)) > longest_caption:
                    longest_caption = len(caption)

        prompt = [images]
        if task_induction is not None:
            task_induction_tokenized = tokenizer.encode(
                task_induction, return_tensors="pt", truncation=True
            ).repeat(images.shape[0], 1)
            prompt.append(task_induction_tokenized)
        prompt = model.embed(prompt)
        model_output = model.generate(
            prompt,
            max_steps=longest_caption + 2,
            temperature=temperature,
        )  # restrict generation to the length of the longest caption, give or take some leeway

        for idx, output in enumerate(model_output):
            if is_main():
                print(prompt.shape)
                print(task_induction)
                print("\nOUTPUT: ")
                print("-" * 80)
                print(output)
                print("-" * 80)
                print("REFERENCES: ")
                print("-" * 80)
                print(captions[idx])
                print("\n")
            references.append(captions[idx])
            hypotheses.append(output)

    if world_size > 1:
        # since different indices are processed across different processes, we need to gather the results
        # and then process them together (BLEU is a corpus-level metric, so we need to do this)
        # we do this in a rather hacky way by saving the results to a file and then loading them back
        # this will only work if all ranks share a common filesystem
        all_paths = [
            Path("/tmp") / f"coco_eval_results_{rank}.json"
            for rank in range(world_size)
        ]
        current_rank_path = all_paths[rank]
        with open(current_rank_path, "w") as f:
            json.dump({"references": references, "hypotheses": hypotheses}, f)

        # torch.distributed.barrier()
        time.sleep(30)  # wait for all ranks to finish writing to the file

        # assert all files exist
        for path in all_paths:
            assert path.exists(), f"Could not find {path}"

        # load all results
        references = []
        hypotheses = []
        for path in all_paths:
            with open(path, "r") as f:
                data = json.load(f)
                references.extend(data["references"])
                hypotheses.extend(data["hypotheses"])

        # torch.distributed.barrier()
        time.sleep(10)  # wait for all ranks to finish reading the file

        # remove all files
        for path in all_paths:
            if is_main():
                path.unlink()

    # right now the references are formatted such that references[0] contains all the references for hypothesis[0]
    # but nlgeval expects them to be formatted such that the references for hypothesis[0] are in references[i] for i in range(n_references)
    # i.e each reference list should have the same number of elements as the hypothesis list
    # so we need to reformat the references
    _references = [[], [], [], [], []]  # each hypothesis has 5 references
    for r in references:
        assert len(r) >= 5, f"length of references is {len(r)}, should be >= 5"
        # no idea why some are longer, this is such a hassle
        # since they are fairly rare, we just take the first 5
        r = r[:5]
        for idx, ref in enumerate(r):
            _references[idx].append(ref)

    metrics_dict = compute_nlg_metrics(references=_references, hypothesis=hypotheses)
    if return_captions:
        captions = [
            {"hypothesis": hyp, "references": references[i]}
            for i, hyp in enumerate(hypotheses)
        ]
        metrics_dict["captions"] = captions
    return metrics_dict