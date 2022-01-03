from torch.utils.data import Dataset
import torch
import json
from pathlib import Path
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools
from torchvision import transforms as T
from PIL import Image
import requests
import random
import traceback


def _maybe_download_image(d, images_dir):
    left_img_url = d["left_url"]
    right_img_url = d["right_url"]
    left_img_path = images_dir / Path(left_img_url).name
    right_img_path = images_dir / Path(right_img_url).name
    for url, img in [(left_img_url, left_img_path), (right_img_url, right_img_path)]:
        try:
            if not img.exists():
                r = requests.get(url, timeout=10)
                if r.status_code == 200:
                    with open(img, "wb") as f:
                        f.write(r.content)
                else:
                    print(f"Failed to download {url}: Status Code {r.status_code}")
                    return 1
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return 1
    return 0


class NLVR2Dataset(Dataset):

    """
    A utility to download + iterate over the data from NVLR2 https://lil.nlp.cornell.edu/nlvr/
    """

    # I have no idea why there are so many splits either
    SPLITS = [
        "balanced_dev",
        "balanced_test",
        "unbalanced_dev",
        "unbalanced_test",
        "dev",
        "val",
        "test",
        "train",
    ]

    URLS = {
        "balanced_dev": "https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/balanced/balanced_dev.json",
        "balanced_test": "https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/balanced/balanced_test1.json",
        "unbalanced_dev": "https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/unbalanced/unbalanced_dev.json",
        "unbalanced_test": "https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/unbalanced/unbalanced_test1.json",
        "dev": "https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/dev.json",
        "val": "https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/dev.json",  # map val to dev
        "test": "https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/test1.json",
        "train": "https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/train.json",
    }

    def __init__(
        self,
        data_dir,
        split,
        tokenizer=None,
        transforms=None,
        seq_len=2048,
        warmup=True,
        return_tensors=True,
        return_img_list=False,
    ):

        self.return_tensors = return_tensors

        if self.return_tensors:
            assert (
                tokenizer is not None and transforms is not None
            ), "tokenizer or transforms missing"

        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._split = split
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.transforms = transforms
        self.failed_indices = []
        self.failed_indices_file = self.data_dir / f"failed_indices_{self.split}.json"
        self.return_img_list = return_img_list

        # load failed indices from disk if they exist
        if self.failed_indices_file.exists():
            with open(self.failed_indices_file, "r") as f:
                self.failed_indices = json.load(f)

        self.data = self.load_data()

        self.filter_failed()
        self.download_images()  # download images if they don't exist and haven't previously failed

        if warmup:
            for _ in tqdm(
                self.data,
                f"filtering out NLVR2 {self.split} split indices that failed to download...",
            ):
                pass  # this iterates through the data one time, and filters out the failed indices
            self.filter_failed()

    @property
    def num_classes(self):
        return 2

    @property
    def split(self):
        # returns the split name
        if self._split in ["dev", "val"]:
            return "dev"
        return self._split

    def filter_failed(self):
        # if previous failed indices exist, filter out the failed indices so we don't try to download them again
        if self.failed_indices:
            self.data = [
                d for i, d in enumerate(self.data) if i not in self.failed_indices
            ]

    def download_data(self):
        """
        Downloads the data for the appropriate split from the github repo
        """
        data_path = self.data_dir / f"{self.split}.json"
        if not data_path.exists():
            print(f"Downloading {self.split} data...")
            os.system(f"wget -O {data_path} {self.URLS[self.split]}")

    def load_data(self):
        """
        Loads the data from the json file, downloads it if it doesn't exist
        """
        data_path = self.data_dir / f"{self.split}.json"
        if not data_path.exists():
            self.download_data()
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f.readlines()]
        return data

    def download_images(self):
        """
        Downloads the images for the appropriate split from the github repo
        """

        pbar = tqdm(
            total=len(self.data), desc=f"Downloading NLVR2 {self.split} set images..."
        )
        with Pool(processes=cpu_count()) as pool:
            for idx, failed in enumerate(
                pool.imap(
                    functools.partial(
                        _maybe_download_image, images_dir=self.images_dir
                    ),
                    self.data,
                )
            ):
                pbar.update()
                if failed:
                    # mark index as failed
                    self.failed_indices.append(idx)

        # cache failed indices to disk
        with open(self.failed_indices_file, "w") as f:
            json.dump(self.failed_indices, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, transforms=False):
        try:
            d = self.data[idx]
            left_img_path = self.images_dir / Path(d["left_url"]).name
            right_img_path = self.images_dir / Path(d["right_url"]).name

            left_img = Image.open(left_img_path)
            right_img = Image.open(right_img_path)
            sentence = d["sentence"]
            label = d["label"]

            if self.return_tensors:
                left_img = self.transforms(left_img)
                right_img = self.transforms(right_img)
                sentence = self.tokenizer.encode(
                    sentence,
                    return_tensors="pt",
                    max_length=self.seq_len,
                    padding="max_length",
                )

                label = torch.tensor(d["label"] == "True").long()

            if self.return_img_list:
                return [left_img, right_img], sentence, label
            else:
                return left_img, right_img, sentence, label

        except Exception as e:
            print(f"Failed to get item {idx}: {e}")
            traceback.print_exc()
            # add index to failed indices
            self.failed_indices.append(idx)
            # cache failed indices to disk
            with open(self.failed_indices_file, "w") as f:
                json.dump(self.failed_indices, f)
            idx = random.randint(0, len(self))
            return self[idx]


def display(output):
    """
    Displays the output of the model
    """
    if output is None:
        print("No output to display")
        return
    left_img, right_img, sentence, label = output

    print(f"Sentence: {sentence}")
    print(f"Label: {label}")

    # save images
    left_img.save(f"aa_left_img.png")
    right_img.save(f"aa_right_img.png")
    print("done")
