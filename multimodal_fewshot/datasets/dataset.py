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

try:
    from vqa_eval import VQADataset, OKVQADataset, GQADataset
    from coco import coco_dataset
except ImportError:
    from .vqa_eval import VQADataset, OKVQADataset, GQADataset
    from .coco import coco_dataset

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


class ClassificationDatasetABC(Dataset, ABC):
    """Abstract class for a classification dataset"""

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of classes"""
        pass

    @abstractmethod
    def getitem(
        self, index
    ) -> Tuple[TensorType["b", "c", "h", "w"], TensorType["b", "s"], TensorType["b"]]:
        """A method called by __getitem__ that returns a torch.Tensor containing the image, a tokenized tensor containing the image caption, and an integer representing the class label"""
        pass

    def __getitem__(self, index):
        image, caption, class_label = self.getitem(index)
        return image, caption, class_label


class ClassificationWrapper(ClassificationDatasetABC):
    """Class for a classification dataset that wraps ImgCptDataset"""

    def __init__(self, img_cpt_dataset: ImgCptDataset, num_classes: int):
        self.dataset = img_cpt_dataset
        self._num_classes = num_classes
        super().__init__()

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return self._num_classes

    def getitem(
        self, index
    ) -> Tuple[TensorType["b", "c", "h", "w"], TensorType["b", "s"], TensorType["b"]]:
        """A method called by __getitem__ that returns a torch.Tensor containing the image, a tokenized tensor containing the image caption, and an integer representing the class label"""
        img, caption = self.dataset[index]
        metadata = self.dataset.data[index]["metadata"]
        class_label = torch.tensor(metadata["class_label"])
        return img, caption, class_label


class CCDataset(Dataset):

    """
    Dataset of images and associated captions from a conceptual-captions-structured directory
    """

    def __init__(
        self,
        main_dir: str,
        transform: Optional[Callable] = None,
        size: int = None,
        workers=8,
    ):

        self.main_dir = main_dir
        self.transform = transform
        self.imgs_cpts = []  # [(img_path, img_caption), ... ]

        if main_dir.endswith("/"):
            main_dir = main_dir[:-1]

        dir_list = list(Path(main_dir).glob("*/*.json"))
        pbar = tqdm(dir_list, "preparing CC dataset")
        with Pool(workers) as p:
            for i in p.imap(read_img_captions, pbar):
                self.imgs_cpts += i
        if size is not None:
            self.imgs_cpts = self.imgs_cpts[:size]

    def __len__(self):
        return len(self.imgs_cpts)

    def __getitem__(self, idx: int) -> Tuple[img, str]:
        try:
            img, caption = self.imgs_cpts[idx]
            img = Image.open(f"{self.main_dir}/{img}").convert("RGB")
            if not self.transform is None:
                img = self.transform(img)
            return img, caption
        except (UnidentifiedImageError, OSError):
            # return random index if image is corrupt
            return self[random.randint(0, len(self))]


class HMDataset(Dataset):

    """
    Dataset of images and associated captions from a hateful-memes-structured directory
    """

    def __init__(
        self,
        main_dir: str = "/data/hateful_memes",
        split: str = "train",
        size: int = None,
    ):
        # download_hm(main_dir)
        self.main_dir = main_dir
        self.imgs_txt_labels = []
        with open(f"{self.main_dir}/{split}.jsonl") as file:
            for line in tqdm(file):
                info = json.loads(line)
                self.imgs_txt_labels.append(
                    (info.get("img"), info.get("text"), info.get("label"))
                )
                if size is not None:
                    if len(self.imgs_txt_labels) >= size:
                        break

        print(f"loaded {len(self)} meme images with associated text and class label")

    def __len__(self):
        return len(self.imgs_txt_labels)

    def __getitem__(self, idx):
        try:
            img, text, label = self.imgs_txt_labels[idx]
            img = Image.open(f"{self.main_dir}/{img}").convert("RGB")
            return img, text, label
        except (UnidentifiedImageError, OSError):
            # return random index if image is corrupt
            return self[random.randint(0, len(self))]


class MultimodalDataset(Dataset):
    """
    Wrapper for dataset that preprocesses images and captions into tensors that can be input
    """

    def __init__(self, img_cpts_dset, tokenizer, transforms=None, seq_len=2048):
        self.ds = img_cpts_dset  # dataset yields (img, caption)
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, caption = self.ds[idx]
        if isinstance(
            caption, list
        ):  # if we have multiple captions, pick one at random
            caption = " ".join(caption)
        if self.transforms is not None:
            img = self.transforms(img)
        return (
            img,
            self.tokenizer.encode(
                caption,
                return_tensors="pt",
                max_length=self.seq_len,
                padding="max_length",
                truncation=True,
            ),
        )


def get_dataset(name, data_dir=None, mode="train"):
    assert mode in ["train", "val", "test", "testdev"]
    if name == "coco":
        return coco_dataset(data_dir, mode=mode)
    elif name == "conceptual_captions":
        return CCDataset(data_dir)
    elif name == "hateful_memes":
        return HMDataset(data_dir)
    elif name == "vqa":
        return VQADataset(data_dir, mode=mode)
    elif name == "okvqa":
        return OKVQADataset(data_dir, mode=mode)
    elif name == "gqa":
        return GQADataset(data_dir, mode=mode)
    else:
        raise ValueError(f"{name} not found")


def collate_fn(batch_data: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    # should be able to replace with this:
    return tuple(
        torch.cat(i) for i in list(zip(*batch_data))
    )  # [(img1, cap1), (img2, cap2), ... ] -> [(img1, img2, ... ), (cap1, cap2, ... )])
    """
    all_images, all_captions = list(
        zip(*batch_data)
    )  # [(img1, caption1), (img2, caption2), ... ] -> [(img1, img2, ... ), (caption1, caption2, ... )]
    return torch.cat(all_images), torch.cat([i[:, :2048] for i in all_captions])


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
