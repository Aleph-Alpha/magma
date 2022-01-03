from pathlib import Path
import requests
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool, cpu_count
from functools import partial
import multiprocessing
import time
import zipfile
import math
import json
import random
import torch.distributed as dist


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n|\r")))
    return data


def is_main():
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def _download(
    url,
    out_dir,
    out_path=None,
    disable_pbar=False,
    chunk_size=1024 * 1024,
    session=None,
):
    if out_path is not None:
        out_path = Path(out_path)
    else:
        out_filename = Path(url).name
        out_path = out_dir / out_filename
    if out_path.exists():
        print(f"{str(out_path)} already exists - skipping download")
        return 1
    if session is None:
        r = requests.get(url, stream=True, timeout=10)
    else:
        r = session.get(url, stream=True, timeout=10)
    total_size_in_bytes = int(r.headers.get("content-length", 0))
    if r.status_code == 200:
        pbar = tqdm(
            total=total_size_in_bytes,
            unit="iB",
            unit_scale=True,
            desc=f"Downloading {url} to {out_path}",
            disable=disable_pbar,
        )
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    pbar.update(len(chunk))
                    f.write(chunk)
        return 1
    else:
        if r.status_code not in [404, 410]:  # expected 404 and 410
            print(f"Download for {url} failed: status code {r.status_code}")
        return 0


def download_mp(urls, out_dir):
    """
    Downloads a list of urls to out_dir using multiprocessing
    """
    urls = list(urls)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    fn = partial(_download, out_dir=out_dir, disable_pbar=False)
    with Pool(cpu_count()) as p:
        _ = list(tqdm(p.imap(fn, urls)))


def resize(image: Image, max_size: int):
    """
    Resizes so shortest edge isn't longer than max_size
    """
    original_size = min(image.size[0], image.size[1])
    if original_size >= max_size:
        if image.size[0] < image.size[1]:
            resized_width = max_size
            resized_height = int(
                round((max_size / float(image.size[0])) * image.size[1])
            )
        else:
            resized_height = max_size
            resized_width = int(
                round((max_size / float(image.size[1])) * image.size[0])
            )

        image = image.resize((resized_width, resized_height), Image.BICUBIC)
    return image


class Counter(object):
    """
    A counter that can be used in a multiprocessing environment.
    """

    def __init__(self, print_every=50):
        self.start_time = None
        self.val = multiprocessing.RawValue("i", 0)
        self.successes = multiprocessing.RawValue("i", 0)
        self.lock = multiprocessing.Lock()
        self.print_every = print_every

    def increment(self, i):
        with self.lock:
            if self.start_time is None:
                self.start_time = time.time()
            self.val.value += 1
            if i:
                self.successes.value += 1
            if self.val.value % self.print_every == 0:
                print("-" * 50)
                images_per_sec = self.val.value / (time.time() - self.start_time)
                print(
                    f"PROGRESS: {self.val.value} images downloaded. SUCCESS RATE: {self.successes.value / self.val.value:.2f} | IM/S {images_per_sec}"
                )
                print("-" * 50)

    @property
    def value(self):
        return self.val.value


def unzip(zip_filepath, target_dir):
    print(f"Unzipping {zip_filepath} to {target_dir}")
    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
        zip_ref.extractall(target_dir)


def round_to_nearest(n, m=10000):
    return math.floor(n / m)


def get_data_parallel_indices(
    rank, world_size, length, shuffle=True, max_n_steps=None, random_seed=42
):
    """
    Returns a list of indices unique to each rank for data-parallel evaluation.
    """
    indices = list(range(length))
    if world_size > 1:
        # get indices for specific rank:
        indices = indices[rank:length:world_size]
    if shuffle:
        random.seed(random_seed)
        random.shuffle(indices)
    if max_n_steps is not None:
        indices = indices[: max_n_steps // world_size]
    return indices
