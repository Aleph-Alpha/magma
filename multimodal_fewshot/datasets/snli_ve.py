from pathlib import Path
import shutil
import os
import os
import jsonlines
from collections import defaultdict, OrderedDict
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple
from torchvision import transforms as T
from torchtyping import TensorType
import torch
import random

try:
    from .dataset_utils import download_mp, unzip
    from .dataset import ClassificationDatasetABC
except:
    from dataset_utils import download_mp, unzip
    from dataset import ClassificationDatasetABC

############################################################################################

"""
SNLI-VE Generator
Authors: Ning Xie, Farley Lai(farleylai@nec-labs.com)
# Copyright (C) 2020 NEC Laboratories America, Inc. ("NECLA"). 
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""


def prepare_all_data(SNLI_root, SNLI_files):
    """
    This function will prepare the recourse towards generating SNLI-VE dataset
    :param SNLI_root: root for SNLI dataset
    :param SNLI_files: original SNLI files, which can be downloaded via
                       https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    :return:
        all_data: a set of data containing all split of SNLI dataset
        image_index_dict: a dict, key is a Flickr30k imageID, value is a list of data indices w.r.t. a Flickr30k imageID
    """
    data_dict = {}
    for data_type, filename in SNLI_files.items():
        filepath = os.path.join(SNLI_root, filename)
        data_list = []
        with jsonlines.open(filepath) as jsonl_file:
            for line in jsonl_file:
                pairID = line["pairID"]
                gold_label = line["gold_label"]
                # only consider Flickr30k (pairID.find('vg_') == -1) items whose gold_label != '-'
                if gold_label != "-" and pairID.find("vg_") == -1:
                    imageId = pairID[
                        : pairID.rfind(".jpg")
                    ]  # XXX Removed suffix: '.jpg'
                    # Add Flikr30kID to the dataset
                    line["Flickr30K_ID"] = imageId
                    line = OrderedDict(sorted(line.items()))
                    data_list.append(line)
        data_dict[data_type] = data_list

    # all_data contains all lines in the original jsonl file
    all_data = data_dict["train"] + data_dict["dev"] + data_dict["test"]

    # image_index_dict = {image:[corresponding line index in data_all]}
    image_index_dict = defaultdict(list)
    for idx, line in enumerate(all_data):
        pairID = line["pairID"]
        imageID = pairID[: pairID.find(".jpg")]
        image_index_dict[imageID].append(idx)

    return all_data, image_index_dict


def _split_data_helper(image_list, image_index_dict):
    """
    This will generate a dict for a data split (train/dev/test).
    key is a Flickr30k imageID, value is a list of data indices w.r.t. a Flickr30k imageID
    :param image_list: a list of Flickr30k imageID for a data split (train/dev/test)
    :param image_index_dict: a dict of format {ImageID: a list of data indices}, generated via prepare_all_data()
    :return: a dict of format {ImageID: a lost of data indices} for a data split (train/dev/test)
    """
    ordered_dict = OrderedDict()
    for imageID in image_list:
        ordered_dict[imageID] = image_index_dict[imageID]
    return ordered_dict


def split_data(
    all_data, image_index_dict, split_root, split_files, SNLI_VE_root, SNLI_VE_files
):
    """
    This function is to generate SNLI-VE dataset based on SNLI dataset and Flickr30k split.
    The files are saved to paths defined by `SNLI_VE_root` and `SNLI_VE_files`
    :param all_data: a set of data containing all split of SNLI dataset, generated via prepare_all_data()
    :param image_index_dict: a dict of format {ImageID: a list of data indices}, generated via prepare_all_data()
    :param split_root: root for Flickr30k split
    :param split_files: Flickr30k split list files
    :param SNLI_VE_root: root to save generated SNLI-VE dataset
    :param SNLI_VE_files: filenames of generated SNLI-VE dataset for each split (train/dev/test)
    """
    print(
        "\n*** Generating data split using SNLI dataset and Flickr30k split files ***"
    )
    with open(os.path.join(split_root, split_files["test"])) as f:
        content = f.readlines()
        test_list = [x.strip() for x in content]
    with open(os.path.join(split_root, split_files["train_val"])) as f:
        content = f.readlines()
        train_val_list = [x.strip() for x in content]
    train_list = train_val_list[:-1000]
    # select the last 1000 images for dev dataset
    dev_list = train_val_list[-1000:]

    train_index_dict = _split_data_helper(train_list, image_index_dict)
    dev_index_dict = _split_data_helper(dev_list, image_index_dict)
    test_index_dict = _split_data_helper(test_list, image_index_dict)

    all_index_dict = {
        "train": train_index_dict,
        "dev": dev_index_dict,
        "test": test_index_dict,
    }
    # # Write jsonl files
    for data_type, data_index_dict in all_index_dict.items():
        print("Current processing data split : {}".format(data_type))
        with jsonlines.open(
            os.path.join(SNLI_VE_root, SNLI_VE_files[data_type]), mode="w"
        ) as jsonl_writer:
            for _, index_list in data_index_dict.items():
                for idx in index_list:
                    jsonl_writer.write(all_data[idx])


def parser(SNLI_VE_root, SNLI_VE_files, data_split):
    """
    This is a sample function to parse SNLI-VE dataset
    :param SNLI_VE_root: root of SNLI-VE dataset
    :param SNLI_VE_files: filenames of each data split of SNLI-VE
    :param choice: data split choice, train/dev/test
    """
    filename = os.path.join(SNLI_VE_root, SNLI_VE_files[data_split])
    with jsonlines.open(filename) as jsonl_file:
        for line in jsonl_file:
            # #######################################################################
            # ############ Items used in our Visual Entailment (VE) Task ############
            # #######################################################################

            # => Flikr30kID can be used to find corresponding Flickr30k image premise
            Flickr30kID = str(line["Flickr30K_ID"])
            # =>  gold_label is the label assigned by the majority label in annotator_labels (at least 3 out of 5),
            # If such a consensus is not reached, the gold label is marked as "-",
            # which are already filtered out from our SNLI-VE dataset
            gold_label = str(line["gold_label"])
            # => hypothesis is the text hypothesis
            hypothesis = str(line["sentence2"])

            yield Flickr30kID, gold_label, hypothesis


############################################################################################


class SNLI_VE_Dataset(ClassificationDatasetABC):
    """
    Downloads + formats the SNLI-VE dataset (https://github.com/necla-ml/SNLI-VE)
    """

    SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
    ALEPH_ALPHA_FLICKR30K_URL = (
        "s3://aleph-alpha34rtgyhu/datasets/flickr_30k_images.zip"
    )

    # SNLI-VE generation resource: SNLI dataset
    SNLI_files = {
        "dev": "snli_1.0_dev.jsonl",
        "test": "snli_1.0_test.jsonl",
        "train": "snli_1.0_train.jsonl",
    }

    SPLIT_FILES = {
        "test": "flickr30k_test.lst",
        "train_val": "flickr30k_train_val.lst",
    }

    SNLI_VE_files = {
        "dev": "snli_ve_dev.jsonl",
        "test": "snli_ve_test.jsonl",
        "train": "snli_ve_train.jsonl",
    }

    ENTAILMENT_TO_INT = {
        "neutral": 0,
        "entailment": 1,
        "contradiction": 2,
    }

    INT_TO_ENTAILMENT = {0: "neutral", 1: "entailment", 2: "contradiction"}

    def __init__(
        self, data_dir, tokenizer, transforms=None, mode="train", seq_len=2048
    ):
        if mode == "val":
            # map val to dev
            mode = "dev"
        assert mode in self.SNLI_VE_files.keys()
        super().__init__()
        self.data_dir = Path(data_dir)
        self.SNLI_root = self.data_dir / "snli_1.0"
        self.data = None
        self.tokenizer = tokenizer
        self.mode = mode

        self.download()
        self.generate_data()
        self.data = list(parser(self.data_dir, self.SNLI_VE_files, mode))

        if transforms is None:
            # default to transform to tensor
            self.transforms = T.ToTensor()
        else:
            self.transforms = transforms

        self.seq_len = seq_len

    @property
    def num_classes(self):
        return 3

    def download(self):

        ## Download SNLI:

        snli_out_path = self.data_dir / "snli_1.0"
        if not snli_out_path.exists():
            download_mp([self.SNLI_URL], self.data_dir)
            snli_zip_path = self.data_dir / "snli_1.0.zip"

            # unzip if not already unzipped
            unzip(snli_zip_path, self.data_dir)
            # remove __MACOSX folder
            try:
                shutil.rmtree(self.data_dir / "__MACOSX")
            except FileNotFoundError:
                pass

            # remove zip file
            snli_zip_path.unlink()

        ## Download Flickr30k:

        flickr_30k_out_path = self.data_dir / "flickr30k_images"
        flickr_30k_zip_path = self.data_dir / "flickr30k_images.zip"
        image_dir_path = self.data_dir / "flickr30k_images" / "flickr30k_images"
        if not flickr_30k_out_path.exists():
            if self.ALEPH_ALPHA_FLICKR30K_URL is None:
                # if we make the repo public, we'll need to remove the s3 url
                raise ValueError(
                    f"Please download Flickr30k Manually to {str(flickr_30k_out_path)} from https://www.kaggle.com/hsankesara/flickr-image-dataset"
                )
            else:
                # download from s3
                aws_cmd = f"aws s3 cp {self.ALEPH_ALPHA_FLICKR30K_URL} {str(flickr_30k_zip_path)}"
                print(aws_cmd)
                os.system(aws_cmd)

                # unzip if not already unzipped
                unzip(flickr_30k_zip_path, self.data_dir)

                # remove zip file
                flickr_30k_zip_path.unlink()

        # download flickr30k test / train split
        test_list_out_path = self.data_dir / "flickr30k_test.lst"
        if not test_list_out_path.exists():
            download_mp(
                [
                    "https://raw.githubusercontent.com/necla-ml/SNLI-VE/master/data/flickr30k_test.lst"
                ],
                self.data_dir,
            )

        train_list_out_path = self.data_dir / "flickr30k_train_val.lst"
        if not train_list_out_path.exists():
            download_mp(
                [
                    "https://raw.githubusercontent.com/necla-ml/SNLI-VE/master/data/flickr30k_train_val.lst"
                ],
                self.data_dir,
            )

    def generate_data(self):
        # if all SNLI-VE files are already generated, skip
        SNLI_VE_Paths = [
            self.data_dir / self.SNLI_VE_files[split] for split in self.SNLI_VE_files
        ]
        if all([p.exists() for p in SNLI_VE_Paths]):
            return

        print("*** SNLI-VE Generation Start! ***")
        all_data, image_index_dict = prepare_all_data(self.SNLI_root, self.SNLI_files)
        split_data(
            all_data,
            image_index_dict,
            self.data_dir,
            self.SPLIT_FILES,
            self.data_dir,
            self.SNLI_VE_files,
        )
        print("*** SNLI-VE Generation Done! ***")

    def load_img(self, flickr30k_id):
        image_path = (
            self.data_dir
            / "flickr30k_images"
            / "flickr30k_images"
            / f"{flickr30k_id}.jpg"
        )
        return Image.open(image_path)

    def getitem(
        self, index
    ) -> Tuple[TensorType["b", "c", "h", "w"], TensorType["b", "s"], TensorType["b"]]:
        assert self.data is not None, "Please load data first!"
        flickr_id, gold_label, hypothesis = self.data[index]
        entailment_int = torch.tensor(self.ENTAILMENT_TO_INT[gold_label]).long()
        try:
            img = self.load_img(flickr_id)
        except Exception as e:
            print(f"Error loading image {flickr_id}")
            print(e)
            print(f"Return random index")
            return self.getitem(random.randint(0, len(self.data) - 1))
        img = self.transforms(img)
        hypothesis = self.tokenizer.encode(
            hypothesis,
            return_tensors="pt",
            max_length=self.seq_len,
            padding="max_length",
            truncation=True
        )
        return img, hypothesis, entailment_int

    def __len__(self):
        assert self.data is not None, "Please load data first!"
        return len(self.data)


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = SNLI_VE_Dataset("/mnt/localdisk/snli_ve", tokenizer=tokenizer)
    print(len(dataset))
    print("done")
