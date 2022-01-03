from PIL import Image
from PIL import UnidentifiedImageError
import os
import json
from pathlib import Path
from tqdm import tqdm
from imagehash import phash

try:
    from .dataset import get_dataset
except ImportError:
    from dataset import get_dataset
import shutil


def save_to_jsons(data_list, target_dir, starting_idx=0):
    pbar = tqdm(
        enumerate(data_list), desc=f"saving {len(data_list)} jsons to {str(target_dir)}"
    )
    for k, data in pbar:
        filename = Path(target_dir) / Path(f"{k+starting_idx}.json")
        with open(filename, "w") as f:
            json.dump(data, f)

    return None


def save_images(img_list, target_dir, mode="mv"):
    for img_path in tqdm(
        img_list,
        desc=f"saving {len(img_list)} images (mode={mode}) to {str(target_dir)}",
    ):
        if mode == "mv":
            shutil.move(img_path, target_dir)
        elif mode == "cp":
            shutil.copy(img_path, target_dir)


def parse_dataset(dataset_type, dataset=None):
    """
    should return an iterator yielding an image path (as Path object) and corresponding image data in a dictionary

    image_path, {"captions": [...], "metadata": {...}, }
    """
    img_data_list = []

    print(f"parsing {dataset_type} dataset")

    if dataset_type == "hateful_memes":
        for img, caption, _ in dataset.imgs_txt_labels:
            img_path = Path(f"{dataset.main_dir}/{img}")
            img_data = {
                "captions": [caption],
                "metadata": {"dataset": f"{dataset_type}"},
            }
            img_data_list.append((img_path, img_data))

    if "vqa" in dataset_type:
        if "train" in dataset_type:
            img_prefix = "COCO_train2014_"
        elif "val" in dataset_type:
            img_prefix = "COCO_val2014_"
        for q_a in dataset.q_a_list:
            img_path = Path(f'{dataset.img_dir}/{img_prefix}{q_a["image_id"]:012}.jpg')

            captions = [
                f'Q: {q_a["question"]} A: {a["answer"]}' for a in q_a["answers"]
            ]
            img_data = {
                "captions": captions,
                "metadata": {"dataset": f"{dataset_type}", "q_a_dict": q_a},
            }
            img_data_list.append((img_path, img_data))

    if "gqa" in dataset_type:
        for q, a, img_path in iter(dataset):
            caption = f"Q: {q} A: {a}"
            img_data = {
                "captions": [caption],
                "metadata": {"dataset": f"{dataset_type}", "question": q, "answer": a},
            }
            img_data_list.append((Path(img_path), img_data))

    if dataset_type == "coco":
        for idx in dataset.ids:
            ann_ids = dataset.coco.getAnnIds(imgIds=idx)
            anns = dataset.coco.loadAnns(ann_ids)
            filename = dataset.coco.imgs[idx]["file_name"]
            img_path = Path(f"{dataset.root}/{filename}")
            captions = [ann["caption"] for ann in anns]
            img_data = {
                "captions": captions,
                "metadata": {"dataset": f"{dataset_type}"},
            }
            img_data_list.append((img_path, img_data))

    return img_data_list


def convert_dataset(
    data_dir,
    dataset_type=None,
    dir_size=10000,
    hash_fn=None,
    mode="mv",
    delete_temp=True,
    ds_iterator=None,
    split="train",
):

    assert mode in ["mv", "cp"]
    data_dir = Path(data_dir)

    # temporary directory to download and store the dataset
    if ds_iterator is None:
        temp_dir = data_dir / "temp"
        os.makedirs(temp_dir, exist_ok=True)
        ds = get_dataset(dataset_type, temp_dir, mode=split)
        ds_iterator = parse_dataset(dataset_type, dataset=ds)
    else:
        temp_dir = None

    # folders for images and corresponding data which is stored in a json file for each image
    os.makedirs(data_dir / "images", exist_ok=True)
    os.makedirs(data_dir / "image_data", exist_ok=True)

    img_data_list = []
    img_path_list = []
    save_img_dir = data_dir / "images" / "0"
    save_data_dir = data_dir / "image_data" / "0"
    num_img_dirs = 0

    # save the new locations of all img files in case some datafiles point to the same image
    new_img_locations = {}

    pbar = tqdm(
        enumerate(ds_iterator),
        desc="converting dataset to standard format...",
    )

    for k, (img_path, data) in pbar:
        img_cpt_data = {}
        # get img data
        img_cpt_data.update(data)

        if str(img_path) in new_img_locations.keys():
            # if filename is in the dictionary, it already has a new location
            new_img_path = new_img_locations[str(img_path)]["new_img_path"]
            img_cpt_data["image_path"] = new_img_path
            if hash_fn is not None:
                img_cpt_data["metadata"]["image_hash"] = new_img_locations[
                    str(img_path)
                ]["hash"]
        else:
            # if file exists in the old location, it will get moved to a new directory
            new_img_path = f"images/{save_img_dir.name}/{img_path.name}"
            img_cpt_data["image_path"] = new_img_path
            new_img_locations[str(img_path)] = {"new_img_path": new_img_path}
            # original location is saved an later saved to the new directory
            img_path_list.append(img_path)

            # if given, apply hash fn
            if hash_fn is not None:
                try:
                    img = Image.open(img_path).convert("RGB")
                    hash_str = str(hash_fn(img))
                    img_cpt_data["metadata"]["image_hash"] = hash_str
                    # save hash so it does not have to be recomputed
                    new_img_locations[str(img_path)]["hash"] = hash_str
                except (UnidentifiedImageError, FileNotFoundError):
                    print("Warning: corrupted or non-existent Image")

        img_data_list.append(img_cpt_data)

        # save images in specified images folder (maximum of dir_size images per folder)
        if (len(img_path_list) % dir_size == 0 and len(img_path_list) > 0) or (
            k == len(ds_iterator) - 1
        ):
            os.makedirs(save_img_dir, exist_ok=True)
            save_images(img_path_list, save_img_dir, mode=mode)
            img_path_list = []
            num_img_dirs += 1
            save_img_dir = data_dir / "images" / f"{num_img_dirs}/"

        # save jdon data in specified image_data folder with consecutive labeling of the json files
        if ((k + 1) % dir_size == 0) or (k == len(ds_iterator) - 1):
            os.makedirs(save_data_dir, exist_ok=True)
            save_to_jsons(
                img_data_list, save_data_dir, starting_idx=max(k + 1 - dir_size, 0)
            )
            # empty path and data lists and update save directories for next saving step
            img_data_list = []
            save_data_dir = data_dir / "image_data" / f"{int((k+1)/dir_size)}/"

    # delete temp folder when done
    if delete_temp and temp_dir is not None:
        os.system(f"rm -r {str(temp_dir)}")


class DatasetEditor:
    def __init__(self, edit_fns):
        self.edit_fns = edit_fns

    def __call__(self, data_dir):
        data_dir = Path(data_dir)
        data_list = data_dir.rglob("*.json")
        pbar = tqdm(data_list, desc="Editing dataset...")
        for data in pbar:
            for fn in self.edit_fns:
                fn(data)


if __name__ == "__main__":

    print("done")
