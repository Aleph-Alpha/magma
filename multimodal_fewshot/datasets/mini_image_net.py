from pathlib import Path
import os
from tqdm import tqdm

from convert_datasets import convert_dataset
from imagehash import phash


def build_dataset(splits_dir, target_dir, split="train"):
    """
    builds the miniImageNet dataset in standard format from a directory containing the train, val and test images (usually ../processed_images)
    """
    source_dir = Path(splits_dir) / split
    target_dir = Path(target_dir)

    assert Path.exists(source_dir)
    os.makedirs(target_dir, exist_ok=True)

    class_list = [path.name for path in Path.glob(source_dir, "*")]

    path_data_list = []

    for class_name in tqdm(class_list, desc="parsing miniImageNet dataset..."):
        for path in Path.glob(source_dir / class_name, "*"):
            path_data_list.append(
                (
                    path,
                    {
                        "captions": [f"This is a {class_name}"],
                        "metadata": {
                            "dataset_type": "mini_image_net",
                            "class": class_name,
                        },
                    },
                )
            )

    convert_dataset(target_dir, hash_fn=phash, mode="cp", ds_iterator=path_data_list)


if __name__ == "__main__":

    build_dataset(
        "/mnt/localdisk/imagenet2012/mini-imagenet-tools/processed_images",
        "/mnt/localdisk/mini_image_net_train_converted",
        split="train",
    )

    build_dataset(
        "/mnt/localdisk/imagenet2012/mini-imagenet-tools/processed_images",
        "/mnt/localdisk/mini_image_net_val_converted",
        split="val",
    )

    build_dataset(
        "/mnt/localdisk/imagenet2012/mini-imagenet-tools/processed_images",
        "/mnt/localdisk/mini_image_net_test_converted",
        split="test",
    )

    print("done")