import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
from PIL.Image import Image as img
from PIL.Image import DecompressionBombError
from PIL import UnidentifiedImageError
import os
import json
from pathlib import Path
from io import BytesIO
import base64

from tqdm import tqdm
from typing import List, Tuple, Generator, Callable, Optional
import random
import torchvision
from multiprocessing import Pool, cpu_count

from abc import ABC
from abc import abstractmethod
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple
from torchtyping import TensorType, patch_typeguard
import traceback


def read_jsonl(filename: str) -> Generator[List, None, None]:
    """
    Iterator over data from a jsonl file
    """
    with open(filename) as file:
        for line in file:
            yield json.loads(line.rstrip("\n|\r"))


def read_img_captions(filename: str) -> List[Tuple[str, str]]:
    """
    Yields image_path, image_caption from cc jsonl files
    """
    img_captions = []
    for item in read_jsonl(filename):
        if not "N/A" in item[-2:]:
            img_captions.append((item[-1], item[-2]))
    return img_captions


def load_json(filename):
    try:
        with open(filename) as f:
            return json.load(f)
    except Exception:
        print(f"ERROR: Error loading json file {filename}")
        traceback.print_exc()


def _read_image_data(data_dir):
    image_data = []
    img_data_dir = data_dir / "image_data"
    paths = _load_paths(data_dir)
    pbar = tqdm(
        paths,
        desc=f"loading dataset from {str(data_dir)}",
    )
    # read data with multiprocessing
    with Pool(cpu_count()) as pool:
        for img_data in pool.imap(load_json, pbar):
            if img_data is not None:
                image_data.append(img_data)
    return image_data


def _load_paths(data_dir, sort=True):
    paths = []
    img_data_dir = data_dir / "image_data"
    for p in tqdm(
        Path(img_data_dir).glob("*/*.json"),
        desc=f"loading dataset paths from {str(data_dir)}",
    ):
        paths.append(p)
    return sorted(paths)


class LazyLoader:
    def __init__(self, data_dir):
        self.paths = _load_paths(data_dir)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = load_json(self.paths[idx])
        if data is None:
            return self[random.randint(0, len(self) - 1)]
        return data


class ImgCptDataset(Dataset):
    """
    Dataset which loads image caption data from our standard format and transforms them into tensors that can be input to the model.
    Images are expected to be stored in data_dir/images, image data in data_dir/image_data and each data item is a json file with format {"image_path": img_path, "captions": [caption1, caption2,...], "metadata":{...}}
    """

    def __init__(
        self, data_dir, tokenizer, transforms, seq_len=2048, load_data_in_memory=False
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.seq_len = seq_len
        self.load_data_in_memory = load_data_in_memory
        if self.load_data_in_memory:
            self.data = _read_image_data(self.data_dir)
        else:
            self.data = LazyLoader(self.data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, idx
    ) -> Tuple[TensorType["b", "c", "h", "w"], TensorType["b", "s"]]:
        img_data = self.data[idx]
        try:
            try:
                img_path = self.data_dir / img_data["image_path"]
            except KeyError as e:
                # if no image path is found, assume path is same as .json, but .jpg
                if not self.load_data_in_memory:
                    p = self.data.paths[idx]
                    img_path = (
                        self.data_dir
                        / "images"
                        / Path(p.parent).name
                        / Path(p.name).with_suffix(".jpg")
                    )
                else:
                    raise e
            img = Image.open(img_path)
            img_tensor = self.transforms(img)
            caption = random.choice(img_data["captions"])
            caption_tensor = self.tokenizer.encode(
                caption,
                return_tensors="pt",
                max_length=self.seq_len,
                padding="max_length",
                truncation=True,
            )
            return img_tensor, caption_tensor
        except (
            UnidentifiedImageError,
            OSError,
            DecompressionBombError,
            IndexError,
        ) as e:
            # return random index if image is corrupt
            print(f"Warning: Could not load image {str(img_path)}")
            return self[random.randint(0, len(self) - 1)]


def collate_fn(batch_data: List[Tuple[torch.Tensor, torch.Tensor]], seq_len=2048):

    all_images, all_captions = list(
        zip(*batch_data)
    )  # [(img1, caption1), (img2, caption2), ... ] -> [(img1, img2, ... ), (caption1, caption2, ... )]
    return torch.cat(all_images), torch.cat([i[:, :seq_len] for i in all_captions])


def pil_to_b64(fp):
    image = Image.open(fp)
    buffered = BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


class dataset_formatter:
    def __init__(
        self, img_cpt_ds, data_formatter, target_dir, hash_fn=None, file_prefix=""
    ):
        self.ds = img_cpt_ds
        self.data_formatter = data_formatter
        self.target_path = Path(target_dir)
        self.hash_fn = hash_fn
        self.file_prefix = file_prefix

        os.makedirs(self.target_path, exist_ok=True)

    def format_and_save(self):
        pbar = tqdm(range(0, len(self.ds)), desc="creating json files from dataset..")
        for k in pbar:
            img_cpt_dict = {}
            img_cpt_dict["metadata"] = {}
            img, cpt, metadata = self.data_formatter(self.ds[k])
            # img_enc = pil_to_b64(img)
            # img_cpt_dict["image"] = img_enc
            img_cpt_dict["captions"] = cpt
            img_cpt_dict["metadata"] = metadata
            # if self.hash_fn is not None:
            #     img_cpt_dict["metadata"]["img_hash"] = self.hash_fn(img_enc)

            file_path = self.target_path / Path(f"{self.file_prefix}{k}.json")

            with open(file_path, "w") as f:
                json.dump(img_cpt_dict, f)
