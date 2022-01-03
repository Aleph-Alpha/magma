from pathlib import Path
from tqdm import tqdm
import json
from torch.utils.data import Dataset
import collections
import random
from typing import Optional
from itertools import chain, zip_longest
from .vqa_eval import few_shot_prompt, normalize
import math
import json
import os

nonsense_words = ["dax", "blicket", "slation", "perpo", "shously"]


class OpenEndedFewShot(Dataset):
    def __init__(
        self,
        img_cpt_dataset,
        few_shot_examples=3,
        question_prompt="",
        answer_prompt="",
        separator: str = " ",
        task_induction=False,
        punctuation: str = "",
        repeats=0,
        true_names=False,
    ):
        self.true_names = true_names
        if self.true_names:
            json_path = img_cpt_dataset.data_dir / "label_to_human.json"
            assert Path.exists(json_path)
            with open(json_path) as json_file:
                self.class_names = json.load(json_file)
                self.class_names = {
                    k: v.split(",") for k, v in self.class_names.items()
                }

        self.dataset = img_cpt_dataset

        self.data_dir = self.dataset.data_dir
        self.few_shot_examples = few_shot_examples

        self.tokenizer = self.dataset.tokenizer
        self.transforms = self.dataset.transforms

        self.question_prompt = ""
        self.answer_prompt = ""
        self.separator = ""
        self.task_induction = task_induction
        self.punctuation = punctuation
        self.repeats = repeats

        self.classes = set()
        self.type_to_idx = collections.defaultdict(list)
        for i, item in enumerate(self.dataset.data):
            c = item["metadata"]["class"]
            self.classes.add(c)
            self.type_to_idx[c].append(i)
        self.classes = list(self.classes)

    def get_image_from_id(self, id):
        img_path = self.dataset.data[id]["image_path"]
        return str(Path(self.data_dir) / img_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        c1, c2 = random.sample(self.classes, 2)
        if self.true_names:
            w1 = random.sample(self.class_names[c1], 1)[0].strip()
            w2 = random.sample(self.class_names[c2], 1)[0].strip()
        else:
            w1, w2 = random.sample(nonsense_words, 2)

        # Sample an extra sample for random final task
        image1_ids = random.sample(self.type_to_idx[c1], self.few_shot_examples + 1)
        image2_ids = random.sample(self.type_to_idx[c2], self.few_shot_examples + 1)

        image_paths = []
        answers = []
        questions = []
        for i in range(self.few_shot_examples):
            image_paths += [
                self.get_image_from_id(image1_ids[i]),
                self.get_image_from_id(image2_ids[i]),
            ]
            answers += [" " + w1, " " + w2]
            questions += ["this is a", "this is a"]

        # pick random final task
        final_id = random.randint(0, 1)
        if final_id == 0:
            image_paths.append(self.get_image_from_id(image1_ids[-1]))
            answers.append(" " + w1)
        else:
            image_paths.append(self.get_image_from_id(image2_ids[-1]))
            answers.append(" " + w2)
        questions.append("this is a")

        if self.task_induction:
            task_prompt = f"Answer with {w1} or {w2}."
        else:
            task_prompt = None

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
                task_induction=task_prompt,
                punctuation=self.punctuation,
                repeats=self.repeats,
            ),
            answers[-1],
        )


def open_ended_eval(
    model, prompt, answer, temperature=0.01, max_steps=15, return_acc_only=False
):
    model_output = model.generate(
        model.embed(prompt), max_steps=max_steps, temperature=temperature
    )[
        0
    ]  # restrict generation to 15 steps

    model_output = (
        model_output.lower().split("q:")[0].split("\n")[0]
    )  # cut off answer in case of q: .. a: .. is repeated
    model_output = normalize(
        model_output
    )  # process model output, remove punctuation etc.

    if return_acc_only:
        return model_output == normalize(answer)

    return model_output == normalize(answer), model_output


if __name__ == "__main__":

    from .dataset import ImgCptDataset

    ds = ImgCptDataset("/mnt/localdisk")
    ds = OpenEndedFewShot()
