import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import json
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
import random
from typing import Tuple

MMHS_URL = "http://datasets.cvc.uab.es/MMHS150K/MMHS150K.zip"
HM_DIR = "s3://aleph-alpha34rtgyhu/datasets/hateful_memes/"


def download_mmhs(data_dir="/data/MMHS/"):
    """
    Downloads MMHS if it doesn't exist already
    """
    os.makedirs(data_dir, exist_ok=True)
    all_paths = [
        Path(data_dir) / "MMHS150K_GT.json",
        Path(data_dir) / "img_resized",
        Path(data_dir) / "img_txt",
        Path(data_dir) / "splits",
    ]
    if not all([os.path.exists(p) for p in all_paths]):
        out_path = Path(data_dir) / "MMHS150K.zip"
        os.system(f"wget {MMHS_URL} -P {data_dir}")
        os.system(f"unzip {out_path} -d {data_dir}")


def download_hm(data_dir="/data/hateful_memes/"):
    """
    Downloads Hateful Memes dataset if it doesn't exist already
    """
    os.makedirs(data_dir, exist_ok=True)
    all_paths = [
        Path(data_dir) / "img",
        Path(data_dir) / "train.jsonl",
        Path(data_dir) / "test.jsonl",
        Path(data_dir) / "dev.jsonl",
    ]
    if not all([os.path.exists(p) for p in all_paths]):
        os.system(
            f"aws s3 cp --recursive s3://aleph-alpha34rtgyhu/datasets/hateful_memes {data_dir}"
        )


def clean_tweet(string, *args):
    """
    cleans tweet "@user1 @user2 This is a stupid tweet [img url]" -> "This is a stupid tweet"
    """
    clean_str = string
    if "w" in args:
        clean_str = clean_str.split(" https://t.co")[0]

    if "@" in args:
        while clean_str[0] == "@":
            clean_str = clean_str.split(" ", 1)[1]

    return clean_str


class MMHSDataset(Dataset):
    def __init__(self, data_dir="/data/MMHS/", split="train") -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        download_mmhs(self.data_dir)
        with open(self.data_dir / "MMHS150K_GT.json") as f:
            data = json.load(f)
        self.split = split
        assert split in ["train", "val", "test"]
        with open(self.data_dir / f"splits/{split}_ids.txt") as f:
            self.ids_to_use = [
                i.strip() for i in tqdm(f.readlines(), desc="Loading MMHS split data")
            ]
        # filter out data
        print(f"Filtering MMHS dataset to {split} split")
        self.data = {k: data[k] for k in tqdm(self.ids_to_use, desc="loading metadata")}
        self.image_paths = [
            Path(data_dir) / f"img_resized/{i}.jpg"
            for i in tqdm(self.ids_to_use, "loading image paths")
            if os.path.exists(Path(data_dir) / f"img_resized/{i}.jpg")
        ]
        self.image_texts = {
            i: json.load(open(Path(data_dir) / f"img_txt/{i}.json"))
            for i in tqdm(self.ids_to_use, desc="loading image texts")
            if os.path.exists(Path(data_dir) / f"img_txt/{i}.json")
        }
        print(
            f"loaded {len(self.image_paths)} images, {len(self.data)} metadata and {len(self.image_texts)} OCR'd texts"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path)
        except Exception as e:
            print(f"{path} failed to open - picking a new one at random")
            return self[random.randint(0, len(self) - 1)]
        data = self.data[path.stem]
        data["img_text"] = self.image_texts.get(path.stem, None)
        return (
            image,
            clean_tweet(data["tweet_text"], "w"),
            data["labels"],
        )


# TODO: Also return image text?


def is_pure_label(labels):
    return all([i == 0 for i in labels]) or all(
        [i > 0 for i in labels]
    )  # true if labels agree on hs/no hs


def is_hs_mv(labels):
    return (
        sum([int(bool(i)) for i in labels]) >= len(labels) / 2
    )  # true if majority of labels is hs


def is_hs_sv(labels):
    return not all([i == 0 for i in labels])  # true if at least one label is hs


class MMHSFilter:
    def __init__(self, filter_type):
        self.filter_type = filter_type

    def __call__(self, labels):
        if self.filter_type == "is_pure_label":
            return is_pure_label(labels)
        if self.filter_type == "is_hs_mv":
            return is_hs_mv(labels)
        if self.filter_type == "is_not_hs_mv":
            return not is_hs_mv(labels)
        if self.filter_type == "is_hs_sv":
            return is_hs_sv(labels)
        if self.filter_type == "is_not_hs_sv":
            return not is_hs_sv(labels)


class MMHSDatasetFilter(MMHSDataset):
    def __init__(self, ds, filter_fn=is_hs_mv, negate=False):
        self.data_dir = ds.data_dir
        self.split = ds.split
        self.data = ds.data
        self.image_paths = ds.image_paths
        self.image_texts = ds.image_texts

        pbar = tqdm(self.image_paths, desc="filtering label data")
        if not negate:
            self.image_paths = [
                path for path in pbar if filter_fn(self.data[path.stem]["labels"])
            ]
        else:
            self.image_paths = [
                path for path in pbar if not filter_fn(self.data[path.stem]["labels"])
            ]

    def __len__(self):
        return len(self.image_paths)


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
        download_hm(main_dir)
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


class MultimodalMemeDataset(Dataset):
    """
    Wrapper for dataset that preprocesses data of the form (image, text, class label(s)) into tensors that can be input
    """

    def __init__(
        self, meme_ds, tokenizer, transforms=None, pad_text=True, seq_len=2048
    ):
        self.ds = meme_ds  # dataset yields (img, text, class labels)
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_text = pad_text

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img, text, label = self.ds[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = ToTensor()(img)

        if text is not None:
            if self.pad_text:
                text = self.tokenizer.encode(
                    text,
                    return_tensors="pt",
                    max_length=self.seq_len,
                    padding="max_length",
                    truncation=True
                )
            else:
                text = self.tokenizer.encode(text, return_tensors="pt", truncation=True)
        if isinstance(label, list):
            # convert list into list of 0-1 values, where 0 is a vote for no hs and 1 is a vote for hs
            label = [int(bool(l)) for l in label]
            # derive label weight
            p = sum(label) / len(label)
            # final output is a binary probability with artificial batch dim
            label = torch.tensor([1 - p, p]).unsqueeze(dim=0)
        else:
            if label is not None:
                label = torch.tensor([label], dtype=torch.long)
            else:
                label = torch.zeros((1, 2), dtype=torch.float) - 1
        return img, text, label


class HMFewshotDataset(Dataset):
    """
    Dataset wrapper that returns an encoded fewshot prompt populated with hateful meme examples that can be embedded into the latent space by using the embed method of the multimodal transformer
    """

    def __init__(
        self,
        ds,
        tokenizer,
        embed_fn=None,
        ex_ds=None,
        num_examples=3,
        transforms=None,
        seq_len=2048,
        use_text=True,
        question="Q: Is this meme hateful? A:",
    ):
        self.ds = ds  # dataset yields (img, text, class labels)
        self.ex_ds = (
            ex_ds or ds
        )  # if given uses examples from ex_ds for the fewshot prompt
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.embed_fn = embed_fn
        self.seq_len = seq_len
        self.num_examples = num_examples
        self.use_text = use_text

        self.img_prompt = self.tokenizer.encode("Image:", return_tensors="pt", truncation=True)
        self.text_prompt = self.tokenizer.encode("Text:", return_tensors="pt", truncation=True)
        self.question_prompt = self.tokenizer.encode(
            question, return_tensors="pt", max_length=self.seq_len, padding="max_length", truncation=True
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        indices = random.choices(range(len(self.ex_ds)), k=self.num_examples)

        img, text, label = self.ds[idx]
        examples = [self.ex_ds[i] for i in indices]
        prompt = []
        # populate prompt with the examples from ex_ds
        if len(examples) > 0:
            for example in examples:
                img_enc = self.transforms(example[0])
                text_enc = self.tokenizer.encode(example[1], return_tensors="pt", truncation=True)
                label_text = (
                    "This meme is hateful \n"
                    if example[2] == 1
                    else "This meme is not hateful \n"
                )
                label_enc = self.tokenizer.encode(label_text, return_tensors="pt", truncation=True)
                if self.use_text:
                    prompt += [
                        self.img_prompt,
                        img_enc,
                        self.text_prompt,
                        text_enc,
                        label_enc,
                    ]
                else:
                    prompt += [self.img_prompt, img_enc, label_enc]

        # append the meme that should be classified last together with the question prompt
        img_enc = self.transforms(img)
        text_enc = self.tokenizer.encode(text, return_tensors="pt", truncation=True)
        if self.use_text:
            prompt += [
                self.img_prompt,
                img_enc,
                self.text_prompt,
                text_enc,
                self.question_prompt,
            ][: self.seq_len]
        else:
            prompt += [self.img_prompt, img_enc, self.question_prompt]

        if self.embed_fn is not None:
            prompt = self.embed_fn(prompt)[
                :, : self.seq_len, :
            ]  # truncate to sequence length

        return prompt


def get_dataset(name, data_dir):
    if name == "train_mmhs":
        ds = MMHSDataset(f"{data_dir}/MMHS", split="train")
    if name == "val_mmhs":
        ds = MMHSDataset(f"{data_dir}/MMHS", split="val")
    if name == "test_mmhs":
        ds = MMHSDataset(f"{data_dir}/MMHS", split="test")
    if name == "train_hm":
        ds = HMDataset(f"{data_dir}/hateful_memes", split="train")
    if name == "test_hm":
        ds = HMDataset(f"{data_dir}/hateful_memes", split="test")
    if name == "dev_hm":
        ds = HMDataset(f"{data_dir}/hateful_memes", split="dev")

    return ds


def collate_fn(batch_data):
    all_images, all_text, all_labels = list(
        zip(*batch_data)
    )  # [(img1, text1, label1), (img2, text2, label2), ... ] -> [(img1, img2, ... ), (text1, text2, ... ), (label1,label2,...)]
    return torch.cat(all_images), torch.cat(all_text), torch.cat(all_labels)


def get_collate_fn(
    image_prefix, word_embedding, seq_len, device=torch.device("cpu"), dtype=torch.float
):
    image_prefix = image_prefix.to(device)
    word_embedding = word_embedding.to(device)

    def collate_fn(batch_data):
        batch_emb_list = []
        for data in batch_data:
            emb_list = []
            for tensor in data:
                tensor = tensor.to(device)
                if tensor.ndim == 4:
                    emb_list.append(image_prefix(tensor))
                elif tensor.ndim == 2:
                    emb_list.append(word_embedding(tensor))
            emb_tensor = torch.cat(emb_list, dim=1)[:, :seq_len, :]
            batch_emb_list.append(emb_tensor)

        batch_tensor = torch.cat(batch_emb_list, dim=0).to(dtype)
        return batch_tensor

    return collate_fn


def get_rnd_elements(ds, size=4):
    ls = []
    for _ in range(size):
        idx = random.randint(0, len(ds) - 1)
        ls.append(ds[idx])
    return ls


if __name__ == "__main__":

    train_mmhs = MMHSDataset(data_dir="/mnt/localdisk/MMHS", split="train")
    val_mmhs = MMHSDataset(data_dir="/mnt/localdisk/MMHS", split="val")
    test_mmhs = MMHSDataset(data_dir="/mnt/localdisk/MMHS", split="test")

    # train_hs_filter = MMHSDatasetFilter(ds=train_mmhs, filter_fn=is_hs_mv)
    # val_hs_filter = MMHSDatasetFilter(ds=val_mmhs, filter_fn=is_hs_mv)
    # test_hs_filter = MMHSDatasetFilter(ds=test_mmhs, filter_fn=is_hs_mv)

    # ratios_mmhs = {
    #     "train": len(train_hs_filter) / len(train_mmhs),
    #     "val": len(val_hs_filter) / len(val_mmhs),
    #     "test": len(test_hs_filter) / len(test_mmhs),
    # }

    # train_hm = HMDataset(main_dir="/data/hateful_memes", split="train")
    # test_hm = HMDataset(main_dir="/data/hateful_memes", split="test")
    # dev_hm = HMDataset(main_dir="/data/hateful_memes", split="dev")

    # size_train = sum(map(lambda x: x[2], train_hm.imgs_txt_labels))
    # size_test = sum(map(lambda x: x[2] == 1, test_hm.imgs_txt_labels))
    # size_dev = sum(map(lambda x: x[2], dev_hm.imgs_txt_labels))

    # ratios_hm = {
    #     "train": size_train / len(train_hm),
    #     "test": size_test / len(test_hm),
    #     "dev": size_dev / len(dev_hm),
    # }

    # print("fraction of hs labeled data points (by majority vote) in MMHS:")
    # print(ratios_mmhs)
    # print("fraction of hs labeled data points in HM:")
    # print(ratios_hm)

    print("test")
