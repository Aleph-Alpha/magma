# Multimodal Fewshot Model

## Installation

Make sure PyTorch (Ver >= 1.9.0) and Torchvision are installed. See https://pytorch.org/get-started/locally/.
Then install all further requirements by 

```bash
pip install -r requirements.txt
```

## Download checkpoints:
Checkpoints for an early iteration of the model are [here](https://drive.google.com/u/0/uc?id=1EiAY3IcKWmGADaLDzdG25ykQghUwza6L&export=download)

They can also be downloaded with the following command:
```bash
gdown https://drive.google.com/u/0/uc?id=1EiAY3IcKWmGADaLDzdG25ykQghUwza6L&export=download
```


## Inference API:

To test out the model, you can use the inference API provided at `inference_api.py`.
To run it on localhost, in one shell, run:

```bash
MODEL_DIR=models CKPT=/path/to/checkpoint.pt python3 inference_api.py
```

`MODEL_DIR` is where the gptj weights are downloaded to, and `CKPT` path should be the path to the multimodal model weights. You can also optionally provide a `CONFIG_PATH`, but the model weights provided use the `base.yml` config, which is the default. 

When setup is done, while that runs in the background, you can query the API in another process like so:

```python
from inference_api import caption

url = "https://s3-us-west-2.amazonaws.com/ai2-vision/aishwarya/mscoco_images/train2014/COCO_train2014_000000143482.jpg"
prompt = "Q:  When was this communication device invented? A:"
print(caption([url, prompt]))
# >>> The first communication device was invented in the year of 1876.
```

The url should point to a .jpg / .png image, and the prompts should be a string. You can provide an arbitrary list of images / captions in any order to do fewshot prompting.
e.g:

```python
prompt = ["https://www.h-hotels.com/_Resources/Persistent/b0a231fa7959f037b43c6e6583dcefd3898c4a2b/berlin-brandenburger-tor-04-2843x1600.jpg",
            "Q: Where is this? A: Berlin",
            "https://storage.googleapis.com/afs-prod/media/media:003181861445403f903de279acae9914/3000.jpeg",
            "Q: Where is this? A: Tibet",
            "https://cdn.britannica.com/47/194547-050-52813FB0/aerial-view-Cairo-Egypt.jpg",
            "Q: Where is this? A: Egypt",
            "https://www.mexico-mio.de/fileadmin/_processed_/5/f/csm_teotihuacan-box_482dbc1216.jpg",
            "Q: Where is this? A:"]
print(caption(prompt))
# >>> Mexico
```


## Manual Inference

The following code loads the model from a checkpoint, and runs a QA prompt using an image loaded from disk:

```python
from multimodal_fewshot.model import get_multimodal_model
from PIL import Image 

config_path = "configs/base.yml"
ckpt_path = "mp_rank_00_model_states.pt"

model, transforms, tokenizer = get_multimodal_model(config_path, ckpt_path=ckpt_path)

# load image
img = Image.open('test.jpg')
prompt = "Q: How many people are in this image? A:" # when prompting, it's important not to leave a trailing space

# transform / tokenize
inputs = [transforms(img), tokenizer(prompt, return_tensors='pt')]

# embed + generate response
embeddings = model.embed(inputs)
response = model.generate(embeddings)
print(response)
```
