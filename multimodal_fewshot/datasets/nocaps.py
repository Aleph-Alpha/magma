from torch.utils.data import Dataset
import torch
from pathlib import Path
from torchtyping import TensorType
from typing import List, Tuple
import os
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from PIL import Image, UnidentifiedImageError
import random
from functools import partial

"""

Captions for val / test are held out, so we will need to save predicted results to disk then send to an eval server

Annotations format:

{
    "licenses": [],
    "info": {
        "url": "http://nocaps.org",
        "date_created": "2018/11/06",
        "version": "0.1",
        "description": "nocaps validation dataset",
        "contributor": "nocaps team",
        "year": 2018
    },
    "images": [
        {
            "id": 0,
            "open_images_id": "0013ea2087020901",
            "height": 1024,
            "width": 732,
            "coco_url": "https://requestor-proxy.figure-eight.com/figure_eight_datasets/open-images/validation/0013ea2087020901.jpg",
            "file_name": "0013ea2087020901.jpg",
            "license": 0,
            "date_captured": "2018-11-06 11:04:33"
        },
    ...
    ]
}
"""


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    from shutil import which

    return which(name) is not None


def _maybe_download_image(
    image_data,
    image_dir,
):
    image_path = image_dir / image_data["file_name"]
    if not image_path.exists():
        os.system(f"wget -q {image_data['coco_url']} -O {image_path}")


class NoCapsDataset(Dataset):

    URLS = {
        "val": "https://s3.amazonaws.com/nocaps/nocaps_val_image_info.json",
        "test": "https://s3.amazonaws.com/nocaps/nocaps_test_image_info.json",
    }

    def __init__(self, data_dir, transforms, mode="val"):
        assert mode != "train", "train set for NoCaps should be CoCo"
        assert mode in ["val", "test"], "mode should be val or test"

        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / f"images"
        self.mode = mode
        self.data = None
        self.transforms = transforms

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.download()

    def download(self):
        url = self.URLS[self.mode]

        # download json data to data_dir
        out_path = self.data_dir / f"nocaps_{self.mode}.json"
        if not out_path.exists():
            os.system(f"wget {url} -O {out_path}")

        # read json data
        with open(out_path, "r") as f:
            self.data = json.load(f)

        # check if images are downloaded
        download_fn = partial(_maybe_download_image, image_dir=self.image_dir)
        pbar = tqdm(
            desc=f"downloading nocaps {self.mode} set images...",
            total=len(self.data["images"]),
        )
        with Pool(cpu_count()) as p:
            for _ in p.imap_unordered(
                download_fn,
                self.data["images"],
            ):
                pbar.update()

    def __getitem__(self, index) -> Tuple[TensorType["b", "c", "h", "w"], int]:
        img_data = self.data["images"][index]
        try:
            img_path = self.image_dir / img_data["file_name"]
            img_id = img_data["id"]
            return self.transforms(Image.open(img_path)), int(img_id)
        except (UnidentifiedImageError, OSError):
            # return random index if image is corrupt
            print(f"Warning: Could not load image {str(img_path)}")
            return self[random.randint(0, len(self) - 1)]

    def __len__(self) -> int:
        return len(self.data["images"])


####################################################################################################
# Evaluation code from https://github.com/nocaps-org/updown-baseline/blob/d124faa8421e1fb51aacc9a2cf4b3e85b2a15d16/updown/utils/evalai.py#L12

from collections import defaultdict
import json
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional


class NocapsEvaluator(object):
    r"""
    A utility class to submit model predictions on nocaps splits to EvalAI, and retrieve model
    performance based on captioning metrics (such as CIDEr, SPICE).
    Extended Summary
    ----------------
    This class and the training script together serve as a working example for "EvalAI in the
    loop", showing how evaluation can be done remotely on privately held splits. Annotations
    (captions) and evaluation-specific tools (e.g. `coco-caption <https://www.github.com/tylin/coco-caption>`_)
    are not required locally. This enables users to select best checkpoint, perform early
    stopping, learning rate scheduling based on a metric, etc. without actually doing evaluation.
    Parameters
    ----------
    phase: str, optional (default = "val")
        Which phase to evaluate on. One of "val" or "test".
    Notes
    -----
    This class can be used for retrieving metrics on both, val and test splits. However, we
    recommend to avoid using it for test split (at least during training). Number of allowed
    submissions to test split on EvalAI are very less, and can exhaust in a few iterations! However,
    the number of submissions to val split are practically infinite.
    """

    def __init__(self, phase: str = "val"):

        # Constants specific to EvalAI.
        self._challenge_id = 355
        self._phase_id = 742 if phase == "val" else 743

        # check evalai is installed
        if not is_tool("evalai"):
            raise OSError(
                "evalai cli is not installed. Please install it with `pip install evalai` and configure an auth token with `evalai set_token <auth_token> `."
            )

    def evaluate(
        self, predictions: List[Dict], iteration: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        r"""
        Take the model predictions (in COCO format), submit them to EvalAI, and retrieve model
        performance based on captioning metrics.
        Parameters
        ----------
        predictions: List[Prediction]
            Model predictions in COCO format. They are a list of dicts with keys
            ``{"image_id": int, "caption": str}``.
        iteration: int, optional (default = None)
            Training iteration where the checkpoint was evaluated.
        Returns
        -------
        Dict[str, Dict[str, float]]
            Model performance based on all captioning metrics. Nested dict structure::
                {
                    "B1": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-1
                    "B2": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-2
                    "B3": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-3
                    "B4": {"in-domain", "near-domain", "out-domain", "entire"},  # BLEU-4
                    "METEOR": {"in-domain", "near-domain", "out-domain", "entire"},
                    "ROUGE-L": {"in-domain", "near-domain", "out-domain", "entire"},
                    "CIDEr": {"in-domain", "near-domain", "out-domain", "entire"},
                    "SPICE": {"in-domain", "near-domain", "out-domain", "entire"},
                }
        """
        # Save predictions as a json file first.
        _, predictions_filename = tempfile.mkstemp(suffix=".json", text=True)
        with open(predictions_filename, "w") as f:
            json.dump(predictions, f)

        submission_command = (
            f"evalai challenge {self._challenge_id} phase {self._phase_id} "
            f"submit --file {predictions_filename}"
        )

        submission_command_subprocess = subprocess.Popen(
            submission_command.split(),
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # This terminal output will have submission ID we need to check.
        submission_command_stdout = submission_command_subprocess.communicate(
            input=b"N\n"
        )[0].decode("utf-8")

        submission_id_regex = re.search(
            "evalai submission ([0-9]+)", submission_command_stdout
        )
        try:
            # Get an integer submission ID (as a string).
            submission_id = submission_id_regex.group(0).split()[-1]  # type: ignore
        except:
            # Very unlikely, but submission may fail because of some glitch. Retry for that.
            return self.evaluate(predictions)

        if iteration is not None:
            print(
                f"Submitted predictions for iteration {iteration}, submission id: {submission_id}."
            )
        else:
            print(f"Submitted predictions, submission_id: {submission_id}")

        # Placeholder stdout for a pending submission.
        result_stdout: str = "The Submission is yet to be evaluated."
        num_tries: int = 0

        # Query every 10 seconds for result until it appears.
        while "CIDEr" not in result_stdout:

            time.sleep(10)
            result_stdout = subprocess.check_output(
                ["evalai", "submission", submission_id, "result"]
            ).decode("utf-8")
            num_tries += 1

            # Raise error if it takes more than 5 minutes.
            if num_tries == 30:
                raise ConnectionError(
                    "Unable to get results from EvalAI within 5 minutes!"
                )

        # Convert result to json.
        metrics = json.loads(result_stdout)

        # save results to a file
        with open(f"no_caps_{submission_id}.json", "w") as f:
            json.dump(metrics, f)

        # keys: {"in-domain", "near-domain", "out-domain", "entire"}
        # In each of these, keys: {"B1", "B2", "B3", "B4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"}
        metrics = {
            "in-domain": metrics[0]["in-domain"],
            "near-domain": metrics[1]["near-domain"],
            "out-domain": metrics[2]["out-domain"],
            "entire": metrics[3]["entire"],
        }

        # Restructure the metrics dict for better tensorboard logging.
        # keys: {"B1", "B2", "B3", "B4", "METEOR", "ROUGE-L", "CIDEr", "SPICE"}
        # In each of these, keys: keys: {"in-domain", "near-domain", "out-domain", "entire"}
        flipped_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        for key, val in metrics.items():
            for subkey, subval in val.items():
                flipped_metrics[subkey][key] = subval

        return flipped_metrics


def nocaps_eval(
    model,
    data_dir,
    transforms,
    tokenizer,
    mode="val",
    temperature=0.01,
    max_n_steps=100,  # maximum data samples to process
    max_steps=50,  # maximum tokens generated per sample
    task_induction=None,
    **kwargs,
):
    # check if model was training
    model_was_training = model.training
    model.eval()

    dataset = NoCapsDataset(data_dir=data_dir, transforms=transforms, mode=mode)
    predictions = []

    for image, image_id in tqdm(dataset, "evaluating nocaps..."):
        image = image.cuda().half()
        prompt = [image]
        if task_induction is not None:
            task_induction_tokenized = tokenizer.encode(
                task_induction, return_tensors="pt", truncation=True
            ).repeat(image.shape[0], 1)
            prompt.append(task_induction_tokenized)
        model_output = model.generate(
            model.embed(prompt), max_steps=max_steps, temperature=temperature
        )[0]
        model_output = model_output.strip()
        predictions.append({"image_id": image_id, "caption": model_output})
        print(model_output)

    evaluator = NocapsEvaluator(phase=mode)
    metrics = evaluator.evaluate(predictions)

    # if model was training, return to training
    if model_was_training:
        model.train()
    return metrics


if __name__ == "__main__":
    import sys

    # change path to be outside package
    sys.path.append("/home/opc/sid/multimodal_fewshot")

    from multimodal_fewshot.model import get_multimodal_model

    model, transforms, tokenizer = get_multimodal_model(
        "/home/opc/sid/multimodal_fewshot/configs/ablations/base.yml",
        model_dir="/mnt/localdisk/models",
        ckpt_path="/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_base_2/global_step17500/mp_rank_00_model_states.pt",
        tokenizer_name="gpt2",
    )
    model = model.cuda().half()
    metrics = nocaps_eval(
        model, "/mnt/shared_vol/nocaps", transforms, tokenizer, mode="val"
    )
    print("done")
