from torchvision.datasets import CocoCaptions
from pathlib import Path
from tqdm import tqdm
from convert_datasets import convert_dataset
from imagehash import phash

# NEEDS 2014 Version of COCO!!


def build_dataset(
    coco_dir="/mnt/localdisk/vqa",
    ann_dir="/mnt/localdisk/foil/",
    target_dir="/mnt/localdisk/",
    mode="train",
):
    assert mode in ["train", "val"]
    img_dir = Path(coco_dir) / f"{mode}2014"
    target_dir = Path(target_dir) / f"foil_{mode}_converted"
    ann_suffix = "train" if mode == "train" else "test"
    ann_file = Path(ann_dir) / f"foilv1.0_{ann_suffix}_2017.json"
    coco_ds = CocoCaptions(img_dir, ann_file)

    path_data_list = []

    for idx in tqdm(coco_ds.ids, desc="parsing dataset..."):
        ann_ids = coco_ds.coco.getAnnIds(imgIds=idx)
        anns = coco_ds.coco.loadAnns(ann_ids)
        for ann in anns:
            filename = coco_ds.coco.imgs[idx]["file_name"]
            img_path = Path(f"{coco_ds.root}/{filename}")
            caption = [ann["caption"]]
            class_label = int(ann["foil"])
            target_word = ann["target_word"]
            foil_word = ann["foil_word"]
            path_data_list.append(
                (
                    img_path,
                    {
                        "captions": caption,
                        "metadata": {
                            "dataset_type": f"foil_{mode}",
                            "class_label": class_label,
                            "target_word": target_word,
                            "foil_word": foil_word,
                        },
                    },
                )
            )

    convert_dataset(target_dir, hash_fn=phash, mode="cp", ds_iterator=path_data_list)


if __name__ == "__main__":
    # build_dataset(mode="train")
    # build_dataset(mode="val")
    print("done")