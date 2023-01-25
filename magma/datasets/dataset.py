import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL.Image import Image as img
from PIL.Image import DecompressionBombError
from PIL import UnidentifiedImageError
import json
from pathlib import Path

from tqdm import tqdm
from typing import List, Tuple, Generator
import random
from multiprocessing import Pool, cpu_count

from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple
from torchtyping import TensorType
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
    def __init__(self, data_dir, few_shot):
        self.few_shot = few_shot
        self.paths = _load_paths(data_dir)

    def __len__(self):
        return len(self.paths)//self.few_shot

    def __getitem__(self, idx):
        all_data = []
        for i in range(self.few_shot):
            data = load_json(self.paths[self.few_shot*idx + i])
            all_data.append(data)
        if data is None:
            return self[random.randint(0, len(self) - 1)]
        return all_data


class ImgCptDataset(Dataset):
    """
    Dataset which loads image caption data from our standard format and transforms them into tensors that can be input to the model.
    Images are expected to be stored in data_dir/images, image data in data_dir/image_data and each data item is a json file with format {"image_path": img_path, "captions": [caption1, caption2,...], "metadata":{...}}
    """

    def __init__(
        self, data_dir, tokenizer, transforms, seq_len=2048, load_data_in_memory=False, few_shot=1
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.seq_len = seq_len
        self.load_data_in_memory = load_data_in_memory
        self.few_shot = few_shot
        if self.load_data_in_memory:
            self.data = _read_image_data(self.data_dir)
        else:
            self.data = LazyLoader(self.data_dir, few_shot)

    def __len__(self):
        return len(self.data)//self.few_shot

    def __getitem__(
        self, idx
    ) -> Tuple[TensorType["b", "c", "h", "w"], TensorType["b", "s"]]:
        if self.load_data_in_memory:
            img_data = self.data[self.few_shot *
                                 idx: (self.few_shot * id) + idx]
        else:
            img_data = self.data[0]
        try:
            try:
                img_paths = [self.data_dir / img["image_path"]
                             for img in img_data]
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
            images = []
            text = ""
            for i, p in enumerate(img_paths):
                img = self.transforms(Image.open(p))

                images.append(img)
                caption = random.choice(img_data[i]["captions"])
                text += f'<|image|> {caption} '

            caption_tensor = self.tokenizer.encode(
                text,
                return_tensors="pt",
                max_length=self.seq_len,
                padding="max_length",
                truncation=True,
            )
            img_tensor = torch.cat(images)

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
