import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import json
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import random
import collections
import re

try:
    from .constants import *
    from .dataset_utils import get_data_parallel_indices
except ImportError:
    from constants import *
    from dataset_utils import get_data_parallel_indices

import numpy as np
from functools import lru_cache
import os
from typing import List, Tuple, Callable, Optional
from tqdm import tqdm

# append main folder to sys path so we can import from main package
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from multimodal_fewshot.utils import get_world_info


def few_shot_prompt(
    questions: List[str],
    answers: List[str],
    image_paths: List[str],
    tokenizer: Callable,
    transforms: Callable = T.ToTensor(),
    question_prompt: str = "Q: ",
    answer_prompt: str = "A: ",
    separator: str = " ",
    punctuation: str = "",
    task_induction: Optional[str] = None,
    repeats=0,
) -> List[torch.Tensor]:
    """
    expects lists of image_paths, corresponding questions and answers
    and returns the prompt for the language model in the format
    [img, Q: question, A: answer,...,img_final, Q: question_final, A: ]
    separator separates the questions and answers
    """
    assert (
        len(questions) == len(answers) == len(image_paths)
    ), "Questions, answers and image paths should all be the same length"
    if task_induction is not None:
        task_induction = tokenizer.encode(
            task_induction, return_tensors="pt", truncation=True
        )
    full_prompt = []
    for idx, (q, a, i) in enumerate(zip(questions, answers, image_paths)):
        img = transforms(Image.open(i))
        question = question_prompt + q
        if idx == len(questions) - 1:
            answer = (
                answer_prompt.strip()
            )  # having a space after A: lowers accuracy greatly
        else:
            answer = answer_prompt + a
        caption = tokenizer.encode(
            question + separator + answer + punctuation, return_tensors="pt"
        )
        example = [img, caption]
        if repeats > 0:
            example = [example] * repeats
        if task_induction is not None:
            example = [task_induction] + example
        full_prompt += example
    return full_prompt


class VQADataset(Dataset):
    def __init__(self, data_dir="/data/vqa", mode="train"):
        print("loading vqa test data")
        self.mode = mode
        assert self.mode in ["train", "val"]
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / f"{self.mode}2014"
        self.q_dir = (
            self.data_dir / f"v2_OpenEnded_mscoco_{self.mode}2014_questions.json"
        )
        self.a_dir = self.data_dir / f"v2_mscoco_{self.mode}2014_annotations.json"
        os.makedirs(self.data_dir, exist_ok=True)
        self.download()
        with open(self.q_dir, "r") as qfile:
            self.q_dict = json.load(qfile)

        with open(self.a_dir, "r") as afile:
            self.a_dict = json.load(afile)

        questions = self.q_dict["questions"]
        annotations = self.a_dict["annotations"]

        self.q_a_list = [
            {
                "question_type": a["question_type"],
                "image_id": q["image_id"],
                "question": q["question"],
                "answers": a["answers"],
            }
            for q, a in zip(questions, annotations)
        ]

        self.q_a_list = [
            {
                "question_type": a["question_type"],
                "image_id": q["image_id"],
                "question_id": q["question_id"],
                "question": q["question"],
                "answers": a["answers"],
            }
            for q, a in zip(questions, annotations)
        ]

        self.question_types = set()
        for q_a in self.q_a_list:
            self.question_types.add(q_a["question_type"])

        self.q_a_by_type = collections.defaultdict(list)
        for q_a in self.q_a_list:
            self.q_a_by_type[q_a["question_type"]].append(q_a)

        print("loaded vqa dataset")

    def download(self):
        string = "Val" if self.mode == "val" else "Train"
        url_img = f"http://images.cocodataset.org/zips/{string.lower()}2014.zip"  # http://images.cocodataset.org/zips/val2014.zip
        url_q = f"https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_{string}_mscoco.zip"  # https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
        url_a = f"https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_{string}_mscoco.zip"  # https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

        img_zip = f"{string.lower()}2014.zip"
        q_zip = f"v2_Questions_{string}_mscoco.zip"
        a_zip = f"v2_Annotations_{string}_mscoco.zip"

        if not all(
            [
                os.path.exists(f"{self.data_dir}/{f}")
                for f in [
                    f"v2_OpenEnded_mscoco_{string.lower()}2014_questions.json",
                    f"v2_mscoco_{string.lower()}2014_annotations.json",
                ]
            ]
        ):
            os.system(
                f"wget  -P {self.data_dir} {url_q} && wget  -P {self.data_dir} {url_a} && cd {self.data_dir} && unzip {q_zip} && unzip {a_zip}"
            )

        if not os.path.exists(f"{self.data_dir}/{string.lower()}2014"):
            os.system(
                f"wget -P {self.data_dir} {url_img} && cd {self.data_dir} && unzip {img_zip}"
            )

    def __len__(self):
        return len(self.q_a_list)

    def __getitem__(self, idx):
        return self.q_a_list[idx]  # [img, question/answer, img, ...]


class OKVQADataset(Dataset):
    def __init__(self, data_dir="/data/okvqa", mode="val"):
        print("loading okvqa test data")
        self.mode = mode
        assert mode in ["val", "train"]
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / f"{self.mode}2014"
        self.q_dir = self.data_dir / f"OpenEnded_mscoco_{mode}2014_questions.json"
        self.a_dir = self.data_dir / f"mscoco_{mode}2014_annotations.json"

        os.makedirs(self.data_dir, exist_ok=True)
        self.download()

        with open(self.q_dir, "r") as qfile:
            self.q_dict = json.load(qfile)

        with open(self.a_dir, "r") as afile:
            self.a_dict = json.load(afile)

        questions = self.q_dict["questions"]
        annotations = self.a_dict["annotations"]

        self.q_a_list = [
            {
                "question_type": a["question_type"],
                "image_id": q["image_id"],
                "question": q["question"],
                "answers": a["answers"],
            }
            for q, a in zip(questions, annotations)
        ]

        self.q_a_list = [
            {
                "question_type": a["question_type"],
                "image_id": q["image_id"],
                "question_id": q["question_id"],
                "question": q["question"],
                "answers": a["answers"],
            }
            for q, a in zip(questions, annotations)
        ]

        self.question_types = set()
        for q_a in self.q_a_list:
            self.question_types.add(q_a["question_type"])

        self.q_a_by_type = collections.defaultdict(list)
        for q_a in self.q_a_list:
            self.q_a_by_type[q_a["question_type"]].append(q_a)

        print("loaded okvqa dataset")

    def download(self):
        if not os.path.exists(self.a_dir):
            os.system(
                f"wget --no-check-certificate https://okvqa.allenai.org/static/data/mscoco_{self.mode}2014_annotations.json.zip -P {str(self.data_dir)} && cd {str(self.data_dir)} && unzip mscoco_{self.mode}2014_annotations.json.zip"
            )
        if not os.path.exists(self.q_dir):
            os.system(
                f"wget --no-check-certificate https://okvqa.allenai.org/static/data/OpenEnded_mscoco_{self.mode}2014_questions.json.zip -P {str(self.data_dir)} && cd {str(self.data_dir)} && unzip OpenEnded_mscoco_{self.mode}2014_questions.json.zip"
            )
        if not os.path.exists(self.img_dir):
            os.system(
                f"wget -- no-check-certificate  http://images.cocodataset.org/zips/{self.mode}2014.zip -P {str(self.data_dir)} && cd {str(self.data_dir)} && unzip {self.mode}2014.zip"
            )

    def __len__(self):
        return len(self.q_a_list)

    def __getitem__(self, idx):
        return self.q_a_list[idx]  # [img, question/answer, img, ...]


class VQAFewShot(Dataset):
    def __init__(
        self,
        vqa_dataset,
        tokenizer,
        img_prefix="COCO_train2014_",
        transforms=None,
        few_shot_examples=3,
        question_prompt="Q: ",
        answer_prompt="A: ",
        separator: str = " ",
        task_induction: Optional[str] = None,
        repeats=0,
    ):
        self.dataset = vqa_dataset
        self.few_shot_examples = few_shot_examples
        self.tokenizer = tokenizer
        self.transforms = transforms or T.ToTensor()
        self.img_prefix = img_prefix

        self.question_prompt = question_prompt
        self.answer_prompt = answer_prompt
        self.separator = separator
        self.task_induction = task_induction
        self.repeats = repeats

    def __len__(self):
        return len(self.dataset)

    def get_i_q_a(self, q_a):
        """
        returns the image path, question and a randomly selected answer from a q_a dictionary
        """
        img_pth = f'{self.dataset.img_dir}/{self.img_prefix}{q_a["image_id"]:012}.jpg'
        question = q_a["question"]
        index = random.randint(0, 9)
        answer = q_a["answers"][index]["answer"]
        return img_pth, question, answer

    def __getitem__(self, idx):
        all_q_a = []
        q_a = self.dataset[
            idx
        ]  # {'question_type': ..., 'image_id': ..., 'question': ..., 'answers': [...]}

        # populate examples, ensure everything is in the same question type
        for _ in range(self.few_shot_examples):
            n_items = len(self.dataset.q_a_by_type[q_a["question_type"]])
            while True:
                random_idx = random.randint(0, n_items - 1)
                new_example = self.dataset.q_a_by_type[q_a["question_type"]][random_idx]
                if new_example["question_id"] != q_a["question_id"]:
                    break
            all_q_a.append(new_example)

        # put original example at the end
        all_q_a.append(q_a)
        # put image paths, questions and answers into separate lists
        img_paths, questions, answers = list(
            zip(*[self.get_i_q_a(q_a) for q_a in all_q_a])
        )
        prompts = few_shot_prompt(
            questions,
            answers,
            img_paths,
            self.tokenizer,
            transforms=self.transforms,
            separator=self.separator,
            question_prompt=self.question_prompt,
            answer_prompt=self.answer_prompt,
            task_induction=self.task_induction,
            repeats=self.repeats,
        )

        return prompts, all_q_a[-1]["answers"]


class GQADataset(Dataset):
    def __init__(self, data_dir="/data/gqa", mode="val"):
        # download if not exist
        self.data_dir = Path(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.download()
        assert mode in ["test", "val", "train", "testdev"]  # testdev = eval
        questions_path = self.data_dir / f"{mode}_balanced_questions.json"
        with open(questions_path) as f:
            self.questions = json.load(f)
        self.question_ids = list(self.questions.keys())

    def download(self):
        if not all(
            [
                os.path.exists(self.data_dir / f)
                for f in [
                    "train_balanced_questions.json",
                    "val_balanced_questions.json",
                    "test_balanced_questions.json",
                ]
            ]
        ):
            os.system(
                f"wget https://nlp.stanford.edu/data/gqa/questions1.2.zip -P {str(self.data_dir)} && cd {str(self.data_dir)} && unzip questions1.2.zip"
            )
        if not os.path.exists(self.data_dir / "images"):
            os.system(
                f"wget https://nlp.stanford.edu/data/gqa/images.zip -P {str(self.data_dir)} && cd {str(self.data_dir)} && unzip images.zip"
            )

    def __len__(self):
        return len(self.question_ids)

    def __getitem__(self, idx) -> Tuple[str, str, str]:
        question_id = self.question_ids[idx]
        question_data = self.questions[question_id]
        image_id = question_data["imageId"]
        question = question_data["question"]
        answer = question_data["answer"]
        # full_answer = question_data['fullAnswer']
        image_path = str(self.data_dir / "images" / f"{image_id}.jpg")
        return question, answer, image_path


class GQAFewShot(Dataset):
    def __init__(
        self,
        tokenizer,
        transforms: Callable = T.ToTensor(),
        data_dir="/data/gqa",
        mode="val",
        n=3,
        question_prompt="Q: ",
        answer_prompt="A: ",
        separator: str = " ",
        task_induction: Optional[str] = None,
        repeats=0,
    ):
        self.dataset = GQADataset(data_dir, mode)
        self.n = n
        self.question_prompt = question_prompt
        self.answer_prompt = answer_prompt
        self.separator = separator
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.task_induction = task_induction
        self.repeats = repeats

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        questions, answers, paths = [], [], []
        question, answer, image_path = self.dataset[idx]
        for _ in range(self.n):
            random_idx = random.randint(0, len(self) - 1)
            while random_idx == idx:
                random_idx = random.randint(0, len(self) - 1)
            q, a, p = self.dataset[random_idx]
            questions.append(q)
            answers.append(a)
            paths.append(p)
        questions.append(question)
        answers.append(answer)
        paths.append(image_path)
        return (
            few_shot_prompt(
                questions,
                answers,
                paths,
                tokenizer=self.tokenizer,
                transforms=self.transforms,
                question_prompt=self.question_prompt,
                answer_prompt=self.answer_prompt,
                separator=self.separator,
                task_induction=self.task_induction,
                repeats=self.repeats,
            ),
            answers[-1],
        )


class VQAFewShotNew(Dataset):
    def __init__(
        self,
        img_cpt_dataset,
        few_shot_examples=3,
        question_prompt="Q: ",
        answer_prompt="A: ",
        separator: str = " ",
        task_induction: Optional[str] = None,
        repeats=0,
    ):
        self.dataset = img_cpt_dataset
        self.q_a_list = [
            img_data.get("metadata", {}).get("q_a_dict")
            for img_data in self.dataset.data
        ]
        self.few_shot_examples = few_shot_examples
        self.data = self.dataset.data

        self.tokenizer = self.dataset.tokenizer
        self.transforms = self.dataset.transforms
        self.data_dir = self.dataset.data_dir

        self.question_prompt = question_prompt
        self.answer_prompt = answer_prompt
        self.separator = separator
        self.task_induction = task_induction
        self.repeats = repeats

        self.idx_to_type = {}
        self.type_to_idx = collections.defaultdict(list)
        for i, item in enumerate(self.dataset.data):
            question_type = (
                item.get("metadata", {}).get("q_a_dict", {}).get("question_type")
            )
            self.idx_to_type[i] = question_type
            self.type_to_idx[question_type].append(i)

    def get_examples_of_type(self, question_type, original_id):
        total_items = len(self.type_to_idx[question_type])
        all_examples = []
        for _ in range(self.few_shot_examples):
            while True:
                random_idx = random.randint(0, total_items - 1)
                new_example = self.type_to_idx[question_type][random_idx]
                img_data = self.data[new_example]
                if (
                    img_data.get("metadata", {}).get("q_a_dict", {}).get("question_id")
                    != original_id
                ):
                    all_examples.append(img_data)
                    break
        return all_examples

    def get_i_q_a(self, img_data):
        img_path = img_data["image_path"]  # relative path
        img_path = str(Path(self.data_dir) / img_path)  # resolve path
        q_a = img_data.get("metadata", {}).get("q_a_dict")
        question = q_a["question"]
        answer = q_a["answers"][random.randint(0, 9)]["answer"]
        return img_path, question, answer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        try:
            img_path = self.data_dir / img_data["image_path"]
            q_a = img_data.get("metadata", {}).get("q_a_dict")
            examples = self.get_examples_of_type(
                q_a["question_type"], q_a["question_id"]
            )

            # put original example at the end
            examples.append(img_data)

            # put image paths, questions and answers into separate lists
            img_paths, questions, answers = list(
                zip(*[self.get_i_q_a(img_data) for img_data in examples])
            )

            prompts = few_shot_prompt(
                questions,
                answers,
                img_paths,
                self.tokenizer,
                transforms=self.transforms,
                separator=self.separator,
                question_prompt=self.question_prompt,
                answer_prompt=self.answer_prompt,
                task_induction=self.task_induction,
                repeats=self.repeats,
            )

            return prompts, examples[-1]["metadata"]["q_a_dict"]["answers"]
        except (UnidentifiedImageError, OSError):
            # return random index if image is corrupt
            print(f"Warning: Could not load image {str(img_path)}")
            return self[random.randint(0, len(self))]


class GQAFewShotNew(Dataset):
    def __init__(
        self,
        img_cpt_dataset,
        few_shot_examples=3,
        question_prompt="Q: ",
        answer_prompt="A: ",
        separator: str = " ",
        task_induction: Optional[str] = None,
        repeats=0,
    ):
        self.dataset = img_cpt_dataset
        self.few_shot_examples = few_shot_examples
        self.data = self.dataset.data

        self.tokenizer = self.dataset.tokenizer
        self.transforms = self.dataset.transforms
        self.data_dir = self.dataset.data_dir

        self.question_prompt = question_prompt
        self.answer_prompt = answer_prompt
        self.separator = separator
        self.task_induction = task_induction
        self.repeats = repeats

    def get_i_q_a(self, img_data):
        img_path = self.data_dir / img_data["image_path"]
        question = img_data["metadata"].get("question")
        answer = img_data["metadata"].get("answer")

        return img_path, question, answer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        questions, answers, paths = [], [], []
        img_path, question, answer = self.get_i_q_a(self.data[idx])
        for _ in range(self.few_shot_examples):
            random_idx = random.randint(0, len(self) - 1)
            while random_idx == idx:
                random_idx = random.randint(0, len(self) - 1)
            i, q, a = self.get_i_q_a(self.data[random_idx])
            questions.append(q)
            answers.append(a)
            paths.append(i)
        questions.append(question)
        answers.append(answer)
        paths.append(img_path)
        return (
            few_shot_prompt(
                questions,
                answers,
                paths,
                tokenizer=self.tokenizer,
                transforms=self.transforms,
                question_prompt=self.question_prompt,
                answer_prompt=self.answer_prompt,
                separator=self.separator,
                task_induction=self.task_induction,
                repeats=self.repeats,
            ),
            answers[-1],
        )


def process_punctuation(in_text):
    out_text = in_text
    # iterate through punctuation characters
    for p in PUNCT:
        if (p + " " in in_text or " " + p in in_text) or (
            re.search(COMMA_STRIP, in_text) != None
        ):
            # if the punctuation is found, preceded or followed by a space, then replace with nothing
            out_text = out_text.replace(p, "")
        else:
            # otherwise, replace with a space
            out_text = out_text.replace(p, " ")
    # strip out periods
    out_text = PERIOD_STRIP.sub("", out_text, re.UNICODE)
    return out_text


def process_digits_and_articles(in_text):
    out_text = []
    temp_text = in_text.lower().split()
    # strip out articles + replace words of digits with digits
    for word in temp_text:
        word = DIGIT_MAP.get(word, word)
        if word not in ARTICLES:
            out_text.append(word)
        else:
            pass
    for word_id, word in enumerate(out_text):
        if word in CONTRACTIONS:
            out_text[word_id] = CONTRACTIONS[word]
    out_text = " ".join(out_text)
    return out_text


def normalize(string):
    string = string.replace("\n", " ")
    string = string.replace("\t", " ")
    string = string.strip()
    string = process_punctuation(string)
    string = process_digits_and_articles(string)
    return string


def vqa_eval(
    model, prompt, answers, temperature=0.01, max_steps=10, return_acc_only=False
):
    model_output = model.generate(
        model.embed(prompt), max_steps=max_steps, temperature=temperature
    )[
        0
    ]  # restrict generation to 10 steps
    model_output = (
        model_output.lower().split("q:")[0].split("\n")[0]
    )  # cut off answer in case of q: .. a: .. is repeated
    model_output = normalize(
        model_output
    )  # process model output, remove punctuation etc.

    # truncate model output to the same number of words as the longest answer
    longest_answer = max([len(answer["answer"].split()) for answer in answers])

    model_output = model_output.split()[:longest_answer]
    model_output = " ".join(model_output)

    # accuracy calculation according to VQA protocol
    accuracies = []
    for answer in answers:
        other_gt_answers = [item for item in answers if item != answer]
        # the answers that match the model output
        matching_answer = [
            item
            for item in other_gt_answers
            if normalize(item["answer"]) == model_output
        ]
        # weight the accuracy by the frequency of this answer? sort of
        acc = min(1, float(len(matching_answer) / 3))
        accuracies.append(acc)

    avg_accuracy = sum(accuracies) / len(accuracies)

    if return_acc_only:
        return avg_accuracy

    return (
        avg_accuracy,
        model_output,
        " / ".join(list(set([i["answer"] for i in answers]))),
    )


def vqa_eval_step(config, model_engine, fewshot_dataset):
    total_acc = []
    model_outputs = []
    gt_answers_list = []
    for i in tqdm(range(config.eval_steps), "Doing vqa eval..."):
        prompt, answers = next(fewshot_dataset)
        acc, model_output, gt_answers = vqa_eval(
            model_engine.module,
            prompt,
            answers,
        )
        model_outputs.append(model_output)
        gt_answers_list.append(gt_answers)
        total_acc.append(acc)

    avg_acc = sum(total_acc) / len(total_acc)
    return avg_acc, model_outputs, gt_answers_list


def gqa_eval(
    model, prompt, answer, temperature=0.01, max_steps=10, return_acc_only=False
):
    model_output = model.generate(
        model.embed(prompt), max_steps=max_steps, temperature=temperature
    )[
        0
    ]  # restrict generation to 10 steps
    model_output = (
        model_output.lower().split("q:")[0].split("\n")[0]
    )  # cut off answer in case of q: .. a: .. is repeated
    model_output = normalize(
        model_output
    )  # process model output, remove punctuation etc.

    # print(f"MODEL OUTPUT: ``{model_output}`` | GT: ``{normalize(answer)}``")

    if return_acc_only:
        return model_output == normalize(answer)

    return model_output == normalize(answer), model_output


def gqa_eval_step(config, model_engine, fewshot_dataset):
    total_acc = []
    model_outputs = []
    gt_answers_list = []
    for i in tqdm(range(config.eval_steps), "Doing gqa eval..."):
        prompt, answer = next(fewshot_dataset)
        result, model_output = gqa_eval(model_engine.module, prompt, answer)
        total_acc.append(result)
        model_outputs.append(model_output)
        gt_answers_list.append(answer)
    avg_acc = sum(total_acc) / len(total_acc)
    return avg_acc, model_outputs, gt_answers_list


def infer_dataset_path(data_dir, vqa_dataset_name, mode="val"):
    # look for the converted dataset
    converted_folder = Path(data_dir) / f"{vqa_dataset_name}_{mode}_converted"
    if converted_folder.exists():
        return str(converted_folder)

    # next look for just data_dir / vqa_dataset_name
    dataset_folder = Path(data_dir) / vqa_dataset_name
    if dataset_folder.exists():
        return str(dataset_folder)

    # for fast vqa we need two supporting datasets,
    # visual_genome_converted / imagenet_full_val_converted
    if vqa_dataset_name == "fast_vqa":
        paths = []

        path = Path(data_dir) / "imagenet_full_val_converted"
        if path.exists():
            paths.append(path)
        else:
            raise ValueError(f"Visual genome dataset not found at {path}")

        path = Path(data_dir) / "visual_genome_converted"
        if path.exists():
            paths.append(path)
        else:
            raise ValueError(f"Visual genome dataset not found at {path}")
        return paths

    return None


def run_vqa_eval(
    model,
    tokenizer,
    transforms,
    vqa_dataset_name,
    data_dir=None,
    dataset_path=None,
    few_shot_examples=0,
    question_prompt="Q: ",
    answer_prompt="A: ",
    separator=" ",
    task_induction=None,
    repeats=0,
    temperature=0.01,
    logits_filter_fn="top_k",  # TODO: make sure this gets passed to the eval function
    max_n_steps=None,
    mode="val",
):
    from .dataset import ImgCptDataset

    # try to infer dataset path from data_dir and dataset_name
    if dataset_path is None:
        dataset_path = infer_dataset_path(data_dir, vqa_dataset_name, mode)
    local_rank, rank, world_size = get_world_info()

    # load dataset
    if vqa_dataset_name in ["vqa", "okvqa"]:
        assert (
            dataset_path is not None
        ), f"No dataset path for {vqa_dataset_name} provided, and could not be inferred."
        base_ds = ImgCptDataset(dataset_path, tokenizer, transforms)
        ds = VQAFewShotNew(
            base_ds,
            few_shot_examples=few_shot_examples,
            question_prompt=question_prompt,
            answer_prompt=answer_prompt,
            separator=separator,
            task_induction=task_induction,
            repeats=repeats,
        )
    elif vqa_dataset_name == "gqa":
        assert (
            dataset_path is not None
        ), f"No dataset path for {vqa_dataset_name} provided, and could not be inferred."
        base_ds = ImgCptDataset(dataset_path, tokenizer, transforms)
        ds = GQAFewShotNew(
            base_ds,
            few_shot_examples=few_shot_examples,
            question_prompt=question_prompt,
            answer_prompt=answer_prompt,
            separator=separator,
            task_induction=task_induction,
            repeats=repeats,
        )
    elif vqa_dataset_name == "vizwiz":
        from .vizwiz import VizWizFewShot

        ds = VizWizFewShot(
            data_dir=str(Path(data_dir) / "vizwiz"),
            tokenizer=tokenizer,
            transforms=transforms,
            few_shot_examples=few_shot_examples,
            question_prompt=question_prompt,
            answer_prompt=answer_prompt,
            separator=separator,
            task_induction=task_induction,
            repeats=repeats,
            mode=mode,
        )
    elif vqa_dataset_name == "mini_image_net":
        from .open_ended_miniimagenet import OpenEndedFewShot, open_ended_eval

        assert (
            dataset_path is not None
        ), f"No dataset path for {vqa_dataset_name} provided, and could not be inferred."
        base_ds = ImgCptDataset(dataset_path, tokenizer, transforms)
        ds = OpenEndedFewShot(
            base_ds,
            few_shot_examples=few_shot_examples,
            question_prompt=question_prompt,
            answer_prompt=answer_prompt,
            separator=separator,
            task_induction=task_induction,
            repeats=repeats,
        )
    elif vqa_dataset_name == "fast_vqa" or vqa_dataset_name == "real_fast_vqa":
        from .fast_vqa import FastVQAFewShot

        print(dataset_path)
        assert isinstance(dataset_path, list)
        base_ds = ImgCptDataset(
            dataset_path[0],
            tokenizer,
            transforms,
        )
        ds = FastVQAFewShot(
            base_ds,
            vg_data_dir=dataset_path[1],
            true_names=vqa_dataset_name == "real_fast_vqa",
        )

    INDICES = get_data_parallel_indices(
        rank, world_size, len(ds), max_n_steps=max_n_steps
    )

    total_acc = []
    if vqa_dataset_name == "fast_vqa":
        eval_fn = gqa_eval
    elif "vqa" in vqa_dataset_name or vqa_dataset_name == "vizwiz":
        eval_fn = vqa_eval
    elif vqa_dataset_name == "mini_image_net":
        eval_fn = open_ended_eval
    else:
        eval_fn = gqa_eval

    for i in tqdm(
        INDICES, desc=f"Running Eval for {vqa_dataset_name}", disable=rank > 0
    ):
        prompt, answers = ds[i]
        acc = eval_fn(
            model, prompt, answers, temperature=temperature, return_acc_only=True
        )
        total_acc.append(acc)

    avg_acc = sum(total_acc) / len(total_acc)

    if world_size > 1:
        # reduce the accuracy across all workers

        avg_acc = torch.tensor(avg_acc).cuda()
        torch.distributed.all_reduce(avg_acc, op=torch.distributed.ReduceOp.SUM)
        avg_acc /= world_size
        avg_acc = avg_acc.item()

    return {"accuracy": avg_acc}
