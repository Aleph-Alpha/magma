# %%
from PIL import Image
import requests
from activations.utils.convert_network import convert_pytorch_model_to_rational
from clip.model import Bottleneck
# from magma.magma import (
#     Magma,
# )
from magma.utils import get_tokenizer
from magma.config import MultimodalConfig
from train import _load_img_cpt_datasets, get_pretraining_datasets
from magma.image_input import ImageInput
from magma.transforms import get_transforms
from magma.image_encoders import get_image_encoder
import clip
import torch
import json
from pathlib import Path
from magma.datasets.convert_datasets import convert_dataset
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
torch.cuda.set_device(1)
torch.set_default_dtype(torch.float16)
IMAGE_SIZE = 384
ENCODER_NAME = 'clip_resnet_large'
INPUT_RESOLUTION = 384
SEQ_LENGTH = 2048


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['MASTER_ADDR'] = 'localhost'
# modify if RuntimeError: Address already in use
os.environ['MASTER_PORT'] = '9994'
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "3"
os.environ['WORLD_SIZE'] = "1"

# model = Magma(
#     '/home/ml-mmeuer/adaptable_magma/fb20-dgx2-configs/Switch_Rational_Relu.yml')
# tokenizer, config, transforms = model.tokenizer, model.config, model.transforms

# config = MultimodalConfig.from_yml(
#     '/home/ml-mmeuer/adaptable_magma/fb20-dgx2-configs/Switch_Rational_Relu.yml'
# )

# train_dataset, eval_dataset = get_pretraining_datasets(
#     config, tokenizer, transforms
# )

# img = train_dataset[0][0]
# text = train_dataset[0][1]

# model = model.to('cuda:3').half()
# img = img.to('cuda:3').half()
# text = text.to('cuda:3')
# model(images=img, captions=text)

encoder, transforms = clip.load("RN50x16", device='cuda:3')
encoder = convert_pytorch_model_to_rational(
    encoder, rational_cuda='cuda:3', approx_func='rational:relu', submodule_class=Bottleneck)
image_name = "pexels-photo-1485637.jpeg"
image_url = f"https://images.pexels.com/photos/1485637/{image_name}?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940"
image = Image.open(requests.get(image_url, stream=True).raw)
image = transforms(image).to('cuda:3', dtype=torch.float16)
images = image[None, :]
encoder.visual(images)

# %%
