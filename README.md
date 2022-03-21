# MAGMA -- Multimodal Augmentation of Generative Models through Adapter-based Finetuning

## Authors

### repo (alphabetical)

Constantin (CoEich), Mayukh (Mayukhdeb), Sid (sdtblck)

### paper

Constantin Eichenberg, Sidney Black, Samuel Weinbach, [Aleph Alpha](https://aleph-alpha.com "Independent AI R&D")

Letitia Parcalabescu, Anette Frank, [Heidelberg University](https://www.cl.uni-heidelberg.de "Computational Linguistics at Heidelberg University")


## Abstract

Large-scale pretraining is fast becoming the norm in Vision-Language (VL) modeling. However, prevailing VL approaches are limited by the requirement for labeled data and the use of complex multi-step pretraining objectives. We present MAGMA - a simple method for augmenting generative language models with additional modalities using adapter-based finetuning. Building on Frozen, we train a series of VL models that autoregressively generate text from arbitrary combinations of visual and textual input. The pretraining is entirely end-to-end using a single language modeling objective, simplifying optimization compared to previous approaches. Importantly, the language model weights remain unchanged during training, allowing for transfer of encyclopedic knowledge and in-context learning abilities from language pretraining. MAGMA outperforms Frozen on open-ended generative tasks, achieving state of the art results on the OKVQA benchmark and competitive results on a range of other popular VL benchmarks, while pretraining on 0.2% of the number of samples used to train SimVLM.

Paper on arXiv: https://arxiv.org/abs/2112.05253

## Examples (via Aleph Alpha playground)

 Photos |  Text & Technical
 --- | --- 
 ![A man covering a woman's eyes to hide a present](examples/magma_present.jpg?raw=true "Example_1") |   ![A hand drawn treasure map](examples/magma_treasure.png?raw=true "Example_3")
![A fallen tree is blocking a road](examples/magma_tree.jpg?raw=true "Example_2")   | ![A software architecture](examples/magma_oracle.png?raw=true "Example_4") 


 ## Model design

![MAGMA model design](examples/model.jpg?raw=true "MAGMA model design") 


## About the repository

In this repository we share the main parts of the codebase for training and inference of our MAGMA VL model. The main use of the repo is for downloading our pretrained weights and interacting with the model. We include a script for data parallel training with Deepspeed for finetuning our models or training a MAGMA model from scratch.

## Installation

Make sure PyTorch (Ver >= 1.9.0) and Torchvision are installed. See https://pytorch.org/get-started/locally/.

You can pip install from the git repository with:

```bash
pip install git+https://github.com/Aleph-Alpha/magma.git
```

Make sure that you also download the config:
```
mkdir configs; wget -O configs/MAGMA_v1.yml https://raw.githubusercontent.com/Aleph-Alpha/magma/add-setup/configs/MAGMA_v1.yml
```

Or if you've cloned the repo, you can install all further requirements by:

```bash
pip install -r requirements.txt
```

## Checkpoint

We also publish the model checkpoint that has been used for the publication. It is hosted on our infrastructure and downloads automatically. It can be downloaded manually here: https://bit.ly/aleph_alpha_magma_download 
	
This checkpoint can also be [played around with on a space](https://huggingface.co/spaces/EleutherAI/magma) managed by [Heath Mitchell](https://github.com/Heath123), [AK](https://mobile.twitter.com/ak92501), and [Stella Biderman](https://stellabiderman.com). (This is a 3rd party space, not managed by Aleph Alpha.)

## Loading a model for inference

Downloads the checkpoint file into `checkpoint_path` if it's not already present.

```python
from magma import Magma
from magma.image_input import ImageInput

model = Magma.from_checkpoint(
    config_path = "configs/MAGMA_v1.yml",
    checkpoint_path = "./mp_rank_00_model_states.pt",
    device = 'cuda:0'
)

inputs =[
    ## supports urls and path/to/image
    ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
    'Describe the painting:'
]

## returns a tensor of shape: (1, 149, 4096)
embeddings = model.preprocess_inputs(inputs)  

## returns a list of length embeddings.shape[0] (batch size)
output = model.generate(
    embeddings = embeddings,
    max_steps = 6,
    temperature = 0.7,
    top_k = 0,
)  

print(output[0]) ##  A cabin on a lake
```

## Converting datasets to our format

To convert an image-caption dataset to our dataset class `magma.datasets.ImgCptDataset`, we suggest:

```python
from magma.datasets.convert_datasets import convert_dataset

def my_dataset_iterator():
    """
    Implement an iterator for your dataset that for every datapoint yields a tuple
    image_path, {"captions": [...], "metadata": {...}, }, where image_path is the path to the image as a Path object, captions is a list of caption strings and metadata is an optional field.
    """

if __name__ == "__main__":
    convert_dataset(data_dir="/target/directory", ds_iterator=my_dataset_iterator())

```

## How to train MAGMA

Run the training with:

```bash
deepspeed train.py --config path_to_my_config
```
To continue training from a deepspeed checkpoint, provide the checkpoint directory in the "load" config parameter.

WARNING: By default, instantiating magma via the init method instead of from_checkpoint loads the pretrained CLIP weights but not the pretrained gpt-j weights. For training MAGMA from scratch, download the gpt-j weights from this repo: https://github.com/finetuneanon/transformers and include them in the state dict after initializing the MAGMA model.