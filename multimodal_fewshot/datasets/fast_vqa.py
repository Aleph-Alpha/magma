from pathlib import Path
import os
import tarfile
import re

from pathlib import Path
import torch
from tqdm import tqdm
import json
from torch.utils.data import Dataset
import collections
import random
from typing import Optional
from copy import deepcopy
import urllib

try:
    from .vqa_eval import few_shot_prompt
    from .dataset_utils import round_to_nearest
    from .dataset import ImgCptDataset, _read_image_data
    from .convert_datasets import convert_dataset
except:
    from vqa_eval import few_shot_prompt
    from dataset_utils import round_to_nearest
    from dataset import ImgCptDataset, _read_image_data
    from convert_datasets import convert_dataset

import json
from multimodal_fewshot.utils import get_tokenizer, is_main
from multimodal_fewshot.transforms import clip_preprocess
from imagehash import phash
import json

nonsense_words = ["dax", "blicket", "slation", "perpo", "shously"]
# we should try to use the words from frozen, but if we need more, here are some
extra_nonsense_words = [
    "juddle",
    "intems",
    "hotion",
    "tuders",
    "modence",
    "proats",
    "nullman",
]
IMAGENET_VAL_URL = "https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/image_classification/imagenet2012_validation_labels.txt"
IMAGENET_VAL_LABELS = (
    urllib.request.urlopen(IMAGENET_VAL_URL).read().decode("utf-8").split("\n")
)


IMAGENET_CLASS_TO_WORD_URL = "https://gist.githubusercontent.com/sdtblck/61454e648c6208473af01a282a7c438f/raw/5136f79ad8a278f8f9189c688431dc1a41f72552/imagenet_label_to_human.json"
IMAGENET_CLASS_TO_WORD = json.loads(
    urllib.request.urlopen(IMAGENET_CLASS_TO_WORD_URL).read().decode("utf-8")
)
IMAGENET_CLASS_TO_WORD = {
    k: v.split(",")[0].strip() for k, v in IMAGENET_CLASS_TO_WORD.items()
}  # take only first name


def path_to_class_name(path):
    name = Path(path).stem
    path_id = int(name.split("_")[-1]) - 1
    return IMAGENET_CLASS_TO_WORD[IMAGENET_VAL_LABELS[path_id]]


def _get_class_mappings(true_names_path):

    if true_names_path is None:
        class_names = IMAGENET_CLASS_TO_WORD
    else:
        with open(true_names_path) as json_file:
            class_names = json.load(json_file)
            class_names = {
                k: [i.strip() for i in v.split(",")] for k, v in class_names.items()
            }

    # also build a reverse mapping
    class_names_reverse = {}
    for k, v in class_names.items():
        class_names_reverse[v] = k
        class_names_reverse[v.lower()] = k
    return class_names, class_names_reverse


class FastVQAFewShot(Dataset):
    def __init__(
        self,
        imagenet_dataset,
        vg_data_dir,
        few_shot_examples=3,
        question_prompt="",
        answer_prompt="",
        separator: str = " ",
        task_induction: Optional[str] = None,
        repeats=0,
        true_names=False,
        true_names_path=None,
    ):
        self.class_names, self.class_names_reverse = _get_class_mappings(
            true_names_path
        )
        self.true_names = true_names
        self.dataset = imagenet_dataset
        self.data_dir = self.dataset.data_dir
        self.few_shot_examples = few_shot_examples

        self.tokenizer = self.dataset.tokenizer
        self.transforms = self.dataset.transforms

        self.question_prompt = ""
        self.answer_prompt = ""
        self.separator = separator
        self.task_induction = task_induction
        self.repeats = repeats

        self.classes = set()
        self.type_to_idx = collections.defaultdict(list)
        for i, item in enumerate(self.dataset.data):
            c = item["metadata"]["class"]
            self.classes.add(c)
            self.type_to_idx[c].append(i)
        self.classes = list(self.classes)

        # build filtered VG
        self.VG_path = Path(vg_data_dir)
        self.filtered_VG_data = self.VG_path / "fast_vqa_image_data"

        if self.filtered_VG_data.exists():
            self.filtered_VG_paths = self.filtered_VG_data.glob("*/*.json")
        else:
            self.filtered_VG_paths = []

        if not self.filtered_VG_paths:
            # build filtered VG
            if is_main():
                VG = _read_image_data(Path(self.VG_path))
                self.VG_filtered = list(
                    filter_vg(
                        tqdm(VG, "Filtering Visual Genome"),
                        list(self.class_names.values()),
                    )
                )
                # save to disk
                self.filtered_VG_data.mkdir(exist_ok=True)
                for n, item in enumerate(self.VG_filtered):
                    parent_dir = self.filtered_VG_data / str(round_to_nearest(n))
                    parent_dir.mkdir(exist_ok=True, parents=True)
                    with open(parent_dir / f"{n}.json", "w") as f:
                        json.dump(item, f)
            torch.distributed.barrier() if torch.distributed.is_initialized() else None
            # load filtered VG from disk
            self.VG_filtered = []
            for path in self.filtered_VG_paths:
                with open(path) as f:
                    self.VG_filtered.append(json.load(f))
        else:
            # load filtered VG from disk
            self.VG_filtered = []
            for path in self.filtered_VG_paths:
                with open(path) as f:
                    self.VG_filtered.append(json.load(f))

    def get_image_from_id(self, id):
        img_path = self.dataset.data[id]["image_path"]
        return str(Path(self.data_dir) / img_path)

    def __len__(self):
        return len(self.VG_filtered)

    def __getitem__(self, idx):
        # first get the classes which are present in the image
        VG_data = self.VG_filtered[idx]
        imagenet_classes = VG_data.get("metadata", {}).get("imagenet_objects", [])
        meta = VG_data["metadata"]
        vg_questions = meta.get("questions", [])
        vg_answers = meta.get("answers", [])

        # assign a unique nonsense word to each class
        _nonsense_words = deepcopy(nonsense_words)
        _extra_nonsense_words = deepcopy(extra_nonsense_words)
        class_to_nonsense = {}
        for c in imagenet_classes:
            if _nonsense_words:
                c_nonsense = _nonsense_words.pop(
                    random.randint(0, len(_nonsense_words) - 1)
                )
            else:
                assert _extra_nonsense_words
                c_nonsense = _extra_nonsense_words.pop(
                    random.randint(0, len(_extra_nonsense_words) - 1)
                )
            class_to_nonsense[c] = c_nonsense

        # if specified, transform the class names in the questions / answers into nonsense words
        patterns = {c: re.compile(c, re.IGNORECASE) for c in imagenet_classes}
        if not self.true_names:
            _vg_questions = []
            _vg_answers = []
            for q, a in zip(vg_questions, vg_answers):
                for c in imagenet_classes:
                    pattern = patterns[c]
                    q = pattern.sub(class_to_nonsense[c], q)
                    a = pattern.sub(class_to_nonsense[c], a)
                _vg_questions.append(q)
                _vg_answers.append(a)
            vg_questions = _vg_questions
            vg_answers = _vg_answers

        w1, w2 = random.sample(imagenet_classes, 2)
        c1, c2 = w1, w2

        if not self.true_names:
            # replace with nonsense words
            w1 = class_to_nonsense[w1]
            w2 = class_to_nonsense[w2]

        # get n_few_shot examples of each class
        image1_ids = self.type_to_idx[c1]
        if len(image1_ids) < self.few_shot_examples:
            print("Not enough images for that class!")
            print("Returning random idx")
            return self[random.randint(0, len(self) - 1)]

        image1_ids = random.sample(image1_ids, self.few_shot_examples)

        image2_ids = self.type_to_idx[c2]

        if len(image2_ids) < self.few_shot_examples:
            print("Not enough images for that class!")
            print("Returning random idx")
            return self[random.randint(0, len(self) - 1)]

        image2_ids = random.sample(image2_ids, self.few_shot_examples)

        # Determine which class comes first in support
        if random.random() < 0.5:
            one_first = True
        else:
            one_first = False

        image_paths = []
        answers = []
        questions = []

        for i in range(self.few_shot_examples):
            im_1 = self.get_image_from_id(image1_ids[i])
            im_2 = self.get_image_from_id(image2_ids[i])
            a1 = " " + w1
            a2 = " " + w2
            image_paths += [im_1, im_2] if one_first else [im_2, im_1]
            answers += [a1, a2] if one_first else [a2, a1]
            questions += ["this is a", "this is a"]

        # append final question from VG
        image_path = self.VG_path / Path(VG_data["image_path"])
        q_idx = random.randint(0, len(vg_questions) - 1)
        question = vg_questions[q_idx]
        answer = vg_answers[q_idx]
        answer = answer.lower()

        # if using nonsense words, replace any occurences of them in the final q / a

        image_paths.append(str(image_path))
        answers.append(" " + answer)
        questions.append(question)

        return (
            few_shot_prompt(
                questions,
                answers,
                image_paths,
                self.tokenizer,
                transforms=self.transforms,
                separator=self.separator,
                question_prompt=self.question_prompt,
                answer_prompt=self.answer_prompt,
                task_induction=self.task_induction,
                repeats=self.repeats,
            ),
            answers[-1].strip(),
        )


def open_ended_eval(
    model, prompt, answer, temperature=0.01, max_steps=15, return_acc_only=False
):
    model_output = model.generate(
        model.embed(prompt), max_steps=max_steps, temperature=temperature
    )[
        0
    ]  # restrict generation to 15 steps

    if return_acc_only:
        return answer.strip(" ") in model_output

    return answer.strip(" ") in model_output, model_output


def filter_vg(data, class_names):
    class_names = set([c.lower() for c in class_names])
    for item in data:
        # get all the objects in the datapoint (from VG annotations)
        objects = (
            item.get("metadata", {})
            .get("visual_genome_metadata", {})
            .get("objects", {})
        )
        if objects is None:
            continue
        objects = objects.get("objects", [])
        if not objects:
            continue
        objects = [
            item
            for sublist in [[name for name in obj.get("names", [])] for obj in objects]
            for item in sublist
        ]

        # filter out objects not in the class_names
        imagenet_objects = set([o for o in objects if o in class_names])
        item["metadata"]["imagenet_objects"] = list(imagenet_objects)

        # item should contain at least two objects that are in the class_names
        if len(imagenet_objects) < 2:
            continue

        # go through and get qa pairs from VG
        questions, answers = [], []
        for qa in (
            item.get("metadata", {})
            .get("visual_genome_metadata", {})
            .get("qa", {})
            .get("qas", [])
        ):
            question, answer = qa.get("question", ""), qa.get("answer", "")
            # check if any of the class names in the image are also contained in the question and answer
            if any([c in question.lower() for c in imagenet_objects]):
                questions.append(question)
                answers.append(answer)

        if not questions or not answers:
            continue

        item["metadata"]["questions"] = questions
        item["metadata"]["answers"] = answers

        yield item


def untar_imagenet(imagenet_tar, output_dir):
    # imagenet is a 2 layered tar - there's an outer tar, then a tar for each class
    # we want to extract the inner tar, then untar each class tar
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(imagenet_tar)
        with tarfile.open(imagenet_tar) as tar:
            tar.extractall(output_dir)
    imagenet_dir = output_dir
    pbar = tqdm(list(imagenet_dir.glob("*.tar")))
    for tar_file in pbar:
        # extract to output_dir
        class_name = tar_file.stem
        class_output_dir = output_dir / class_name
        if not class_output_dir.exists():
            class_output_dir.mkdir()
        tar = tarfile.open(tar_file)
        pbar.set_description(f'Extracting {tar_file} to "{class_output_dir}"')
        tar.extractall(str(class_output_dir))
        tar.close()


def build_dataset_iterator(image_list):
    for path in tqdm(image_list, desc="parsing Imagenet dataset..."):
        class_name = path_to_class_name(path)
        yield (
            path,
            {
                "captions": [f"This is a {class_name}"],
                "metadata": {
                    "dataset_type": "mini_image_net",
                    "class": class_name,
                    "class_id": path_to_class_name(path),
                },
            },
        )


def build_dataset(source_dir, target_dir, split="val"):
    """
    builds the Imagenet dataset in standard format
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    assert Path.exists(source_dir)
    os.makedirs(target_dir, exist_ok=True)

    image_list = list(source_dir.glob("*.JPEG"))

    convert_dataset(
        target_dir,
        hash_fn=phash,
        mode="cp",
        ds_iterator=list(build_dataset_iterator(image_list)),
    )


def download_imagenet(output_dir, split="val"):
    output_dir = Path(output_dir) / split
    if split == "val":
        url = "https://the-eye.eu/eleuther_staging/imagenet/ILSVRC2012_img_val.tar"
    elif split == "train":
        url = "https://the-eye.eu/eleuther_staging/imagenet/ILSVRC2012_img_train.tar"
    elif split == "test":
        url = "https://the-eye.eu/eleuther_staging/imagenet/ILSVRC2012_img_test_v10102019.tar"
    else:
        raise ValueError(f"invalid split: {split}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = output_dir / Path(url).name
    if not output_file.exists():
        print(f"Downloading {split} split to {output_file}")
        os.system(f"wget --no-check-certificate -O {output_file} {url}")
    return output_file


if __name__ == "__main__":
    ############################################################
    # TO PREPARE IMAGENET:
    ############################################################

    # split = "val"
    # imagenet_dir = f"/mnt/localdisk/imagenet2012/mini-imagenet-tools/imagenet"
    # output_dir = f"/mnt/localdisk/imagenet_full_{split}_extracted"
    # converted_output_dir = f"/mnt/localdisk/imagenet_full_{split}_converted"
    # imagenet_tar = download_imagenet(imagenet_dir)
    # untar_imagenet(
    #     imagenet_tar,
    #     output_dir,
    # )
    # build_dataset(
    #     output_dir,
    #     converted_output_dir,
    # )

    ############################################################
    # TO PREPARE FAST-VQA:
    ############################################################

    base_ds = ImgCptDataset(
        "/mnt/localdisk/imagenet_full_val_converted",
        get_tokenizer(),
        clip_preprocess(384),
    )
    ds = FastVQAFewShot(base_ds, vg_data_dir="/mnt/localdisk/visual_genome_converted")

    while True:
        print(ds[random.randint(0, len(ds) - 1)])

    # tar_path = "/mnt/localdisk/imagenet2012/mini-imagenet-tools/imagenet"
    # class_names = json.load(
    #     open("/mnt/localdisk/imagenet2012/mini-imagenet-tools/label_to_human.json")
    # )
    # class_names = {k: v.split(",")[0].strip() for k, v in class_names.items()}

    # visual_genome_path = "/mnt/localdisk/visual_genome_converted"
    # VG = _read_image_data(Path(visual_genome_path))
    # VG_filtered = list(
    #     filter_vg(tqdm(VG, "Filtering Visual Genome"), list(class_names.values()))
    # )

    # def imagenet_iterator(imagenet_path):
    #     for tar_file in Path(imagenet_path).glob("*.tar"):
    #         label = tar_file.name[:-4]  # fname is something like 'n01632458.tar'
    #         with tarfile.open(tar_file) as tar:
    #             for member in tar.getmembers():
    #                 yield Image.open(tar.extractfile(member)), label

    # fast_vqa_dir = Path(f"/mnt/localdisk/visual_genome_converted/fast_vqa_image_data")
    # for n, item in enumerate(VG_filtered):
    #     parent_dir = fast_vqa_dir / str(round_to_nearest(n))
    #     parent_dir.mkdir(exist_ok=True, parents=True)
    #     with open(parent_dir / f"{n}.json", "w") as f:
    #         json.dump(item, f)

    # # iterate through VG filtered, get object / class names
    # # form support with images from minimagenet
    # for image, label in imagenet_iterator(tar_path):
    #     print(image, label)