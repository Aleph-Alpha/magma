import tarfile
import json
from pathlib import Path

try:
    from .dataset_utils import load_jsonl, round_to_nearest
except ImportError:
    from dataset_utils import load_jsonl, round_to_nearest

from imagehash import phash
from PIL import Image, UnidentifiedImageError
import traceback
from tqdm import tqdm
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _convert_one(
    args,
    path,
    mode,
    compute_image_metadata,
    images_per_folder,
    image_out_dir,
    data_out_dir,
    quiet=True,
):
    (idx, img_data) = args
    _, url, caption, relative_fp = img_data
    if relative_fp == "N/A":
        return  # some images e.g failed to download

    image_data_out = {"metadata": {}}
    fp_resolved = path / relative_fp

    # add url
    image_data_out["metadata"]["url"] = url

    # add caption
    image_data_out["captions"] = [caption]

    # add image metadata
    if compute_image_metadata:
        try:
            img = Image.open(fp_resolved)
            image_data_out["metadata"]["image_hash"] = str(phash(img))
            image_data_out["metadata"]["height"] = img.height
            image_data_out["metadata"]["width"] = img.width
        except UnidentifiedImageError:
            if not quiet:
                print(f"Unidentified image: {fp_resolved}")
            return
        except Exception:
            if not quiet:
                print(f"Error computing image metadata for {fp_resolved}: ")
                traceback.print_exc()
            return

    output_subdir = f"{round_to_nearest(idx, images_per_folder):09}"

    data_dir = image_out_dir.parent
    image_subdir = image_out_dir / output_subdir
    image_subdir.mkdir(exist_ok=True, parents=True)

    data_subdir = data_out_dir / output_subdir
    data_subdir.mkdir(exist_ok=True, parents=True)

    image_out_path = image_subdir / f"{idx:09}.jpg"
    data_out_path = data_subdir / f"{idx:09}.json"

    image_data_out['image_path'] = str(image_out_path.relative_to(data_dir))

    # depending on the mode, move or copy the image file to the output directory
    if mode == "move":
        shutil.move(fp_resolved, image_out_path)
    elif mode == "copy":
        shutil.copy(fp_resolved, image_out_path)
    elif mode == "dryrun":
        if not quiet:
            print(f"{fp_resolved} -> {image_out_path}")

    # write the image_data json to the output directory
    if mode != "dryrun":
        with open(data_out_path, "w") as f:
            json.dump(image_data_out, f)
    elif not quiet:
        print(f"{data_out_path}")


def convert_cc3m(
    path, mode, out_dir, compute_image_metadata=True, images_per_folder=10000
):
    path = Path(path)
    out_dir = Path(out_dir)
    image_out_dir = out_dir / "images"
    image_out_dir.mkdir(exist_ok=True, parents=True)
    data_out_dir = out_dir / "image_data"
    data_out_dir.mkdir(exist_ok=True, parents=True)

    # first load in *all* the data
    all_data = []
    for fp in path.glob("*/*.json"):
        data = load_jsonl(fp)
        all_data.extend(data)

    # then we want to reformat it into:
    #   root
    #       images
    #           0001
    #               00001.jpg
    #               00002.jpg
    #               ...
    #           ...
    #       image_data
    #           0001
    #               00001.json
    #               00002.json
    #
    # where each individual image_data json is structured like so:
    # {
    #     "captions": [
    #         "caption_1",
    #         "caption_2",
    #         ...
    #     ],
    #     "metadata": {
    #         "url": "https://jnswire.s3.amazonaws.com/jns-media/34/7a/115630/GITEX.jpg",
    #         "height": 760.0,
    #         "width": 505.0,
    #         "image_hash": "def325895227a951"
    #     },
    #     "image_path": "images/000000160/001602223.jpg" # relative to root
    # }

    fn = partial(
        _convert_one,
        path=path,
        mode=mode,
        compute_image_metadata=compute_image_metadata,
        images_per_folder=images_per_folder,
        image_out_dir=image_out_dir,
        data_out_dir=data_out_dir,
    )
    # now we can run the conversion in parallel
    with Pool(cpu_count()) as p:
        pbar = tqdm(total=len(all_data), desc="Converting CC3M")
        for _ in p.imap(fn, enumerate(all_data)):
            pbar.update()
        pbar.close()