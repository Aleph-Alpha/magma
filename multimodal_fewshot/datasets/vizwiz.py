from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from pprint import pprint
import random
from .vqa_eval import few_shot_prompt
from .dataset_utils import _download, unzip, is_main
import torch.distributed as dist


class VizWizDataset(Dataset):

    SPLITS = ["train", "val", "test"]
    URLS = {
        "train": "https://ivc.ischool.utexas.edu/VizWiz_final/images/train.zip",
        "val": "https://ivc.ischool.utexas.edu/VizWiz_final/images/val.zip",
        "test": "https://ivc.ischool.utexas.edu/VizWiz_final/images/test.zip",
        "annotations": "https://ivc.ischool.utexas.edu/VizWiz_final/vqa_data/Annotations.zip",
    }

    def __init__(
        self,
        data_dir,
        tokenizer,
        transforms,
        mode="train",
        load_images=False,
        return_img_cpt=False,
        seq_len=2048,
    ):

        self.data_dir = Path(data_dir)

        self.tokenizer = tokenizer
        self.transforms = transforms
        self.mode = mode
        self.load_images = load_images
        self.return_img_cpt = return_img_cpt
        self.seq_len = seq_len

        # Make data directories if they don't exist
        self.data_dir.mkdir(exist_ok=True, parents=True)

        # Download data if it doesn't exist
        self.download()

        # load in annotations for the split
        annotation_path = self.data_dir / f"Annotations" / f"{self.mode}.json"
        with open(annotation_path, "r") as f:
            self.annotations = json.load(f)

    def download(self):
        split_url = self.URLS[self.mode]
        out_path = self.data_dir / f"{self.mode}.zip"
        unzipped_out_path = self.data_dir / f"{self.mode}"

        # download and unzip the data if it doesn't exist
        if not unzipped_out_path.exists():
            if is_main():
                _download(split_url, out_dir=None, out_path=out_path)
                unzip(out_path, self.data_dir)
                # remove the zip file
                out_path.unlink()

            if dist.is_initialized():
                dist.barrier()

        anns_out_path = self.data_dir / f"Annotations.zip"
        anns_unzipped_out_path = self.data_dir / f"Annotations"

        # download and unzip the annotations if it doesn't exist
        if not anns_unzipped_out_path.exists():
            if is_main():
                _download(
                    self.URLS["annotations"], out_dir=None, out_path=anns_out_path
                )
                unzip(anns_out_path, self.data_dir)
                # remove the zip file
                anns_out_path.unlink()

            if dist.is_initialized():
                dist.barrier()

    def load_img(self, filename):
        try:
            img_path = self.data_dir / f"{self.mode}" / filename
            img = Image.open(img_path)
            if self.transforms is not None:
                img = self.transforms(img)
            return img
        except Exception as e:
            print(f"Error loading image {filename}")
            print(e)
            return None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx, return_all_answers=False):
        anns = self.annotations[idx]
        if self.load_images:
            img = self.load_img(anns["image"])
        else:
            img = self.data_dir / f"{self.mode}" / anns["image"]
        if img is None:
            print(f"Error loading image {anns['image']}")
            return None, {}

        question = anns["question"]

        if self.mode in ["val", "train"]:
            # test set has no answer
            answers = anns["answers"]
            if not return_all_answers:
                answer = random.choice(answers)["answer"]
            else:
                answer = answers
            # confidence = answer["answer_confidence"]  # TODO: return confidence?
        else:
            answer = None

        if self.return_img_cpt:
            assert self.tokenizer is not None
            caption = "Q: " + question + " A: " + answer
            caption = self.tokenizer.encode(
                caption,
                return_tensors="pt",
                max_length=self.seq_len,
                padding="max_length",
            )

            return img, caption

        if self.tokenizer is not None:
            question = self.tokenizer.encode(question, return_tensors="pt")
            if answer is not None:
                answer = self.tokenizer.encode(answer, return_tensors="pt")

        return img, question, answer


class VizWizFewShot(VizWizDataset):
    def __init__(
        self,
        data_dir,
        tokenizer,
        transforms,
        few_shot_examples=3,
        question_prompt="Q: ",
        answer_prompt="A: ",
        separator=" ",
        task_induction=None,
        repeats=0,
        mode="val",
    ):
        super().__init__(
            data_dir,
            tokenizer=None,
            transforms=transforms,
            mode=mode,
            load_images=False,
        )
        self.few_shot_examples = few_shot_examples
        self.question_prompt = question_prompt
        self.answer_prompt = answer_prompt
        self.separator = separator
        self.task_induction = task_induction
        self.repeats = repeats
        self._tokenizer = tokenizer
        assert self.mode in [
            "train",
            "val",
        ], "VizWiz test set has no answers and so can't be used for few shot"

    def __getitem__(self, idx):
        questions, answers, paths = [], [], []
        img_path, question, all_answers = super().__getitem__(
            idx, return_all_answers=True
        )
        for _ in range(self.few_shot_examples):
            random_idx = random.randint(0, len(self) - 1)
            while random_idx == idx:
                random_idx = random.randint(0, len(self) - 1)
            i, q, a = super().__getitem__(random_idx)
            questions.append(q)
            answers.append(a)
            paths.append(i)
        questions.append(question)
        answers.append(None)  # last answer shouldn't be used
        paths.append(img_path)
        return (
            few_shot_prompt(
                questions,
                answers,
                paths,
                tokenizer=self._tokenizer,
                transforms=self.transforms,
                question_prompt=self.question_prompt,
                answer_prompt=self.answer_prompt,
                separator=self.separator,
                task_induction=self.task_induction,
                repeats=self.repeats,
            ),
            all_answers,
        )
