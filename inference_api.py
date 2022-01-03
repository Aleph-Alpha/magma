from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal
import torch

from multimodal_fewshot import get_multimodal_model
from IPython.display import Image
import os
from PIL import Image
from pathlib import Path
import requests

app = FastAPI()
PORT = os.getenv("CAPTIONING_PORT", 1234)
HOST = os.getenv("CAPTIONING_HOST", "0.0.0.0")


class Request(BaseModel):
    """
    Inputs should be a list of image urls / strings
    """

    inputs: List[str]
    temperature: float = 0.01


global_vars = {}


def norm(img):
    low, high = img.min(), img.max()
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))
    return img


def image_tensor_from_url(url, transforms, cache_dir="./images"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    name = str(Path(url).name)
    if (
        not name.lower().endswith(".jpg")
        or name.lower().endswith(".png")
        or name.lower().endswith(".jpeg")
    ):
        name += ".jpg"
    name = name[:50]
    img_path = cache_dir / name
    if not img_path.exists():
        os.system(f"wget --no-check-certificate '{url}' -O {img_path}")
    img = Image.open(img_path)
    return transforms(img)


def fewshot_prompt(inputs: List[str]):
    """
    inputs should be a list of strings, either urls (embedded as images), or captions (embedded as text)
    """
    tokenizer = global_vars["tokenizer"]
    transforms = global_vars["transforms"]
    inputs_encoded = []
    for inp in inputs:
        if inp.startswith("http://") or inp.startswith("https://"):
            inputs_encoded.append(image_tensor_from_url(inp, transforms=transforms))
        else:
            inputs_encoded.append(tokenizer.encode(inp, return_tensors="pt", truncation=True))
    return inputs_encoded


def generate(inputs, temperature=0.001, remove_tokens_after_eos=True):
    # temp / top_p / top_k
    # in future: logit weighting
    model = global_vars["model"]
    prompt = fewshot_prompt(inputs)
    input_embeddings = model.embed(prompt)
    print("-" * 50)
    outputs = model.generate(
        input_embeddings,
        temperature=temperature,
        remove_tokens_after_eos=remove_tokens_after_eos,
    )[0]
    print(outputs)
    print("-" * 50)
    return outputs


@app.on_event("startup")
async def startup_event():
    MODEL_DIR = os.getenv(
        "MODEL_DIR", None
    )  # this is where the gptj model will be downloaded e.g /mnt/localdisk/models
    CKPT = os.getenv(
        "CKPT", None
    )  # this is the path to the multimodal weights, i.e /mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16/global_step27500/mp_rank_00_model_states.pt
    CONFIG_PATH = os.getenv("CONFIG", None)
    # make sure the env vars are set
    assert (
        MODEL_DIR is not None
    ), "MODEL_DIR not set - please run the app with MODEL_DIR=/path/to/dir as an env var. This is where GPTJ weights will be downloaded."
    assert (
        CKPT is not None
    ), "CKPT not set - please run the app with CKPT=/path/to/model/weights.pt "
    CONFIG_PATH = CONFIG_PATH or os.path.join("configs", "base.yml")
    model, transforms, tokenizer = get_multimodal_model(
        config_path=CONFIG_PATH,
        model_dir=MODEL_DIR,
        ckpt_path=CKPT,
    )

    model.cuda().half()
    model.eval()
    global_vars["model"] = model
    global_vars["transforms"] = transforms
    global_vars["tokenizer"] = tokenizer


@app.get("/")  # 0.0.0.0:1234/
async def ping():
    print("hello world")
    return {"output": "hello world"}


@app.get("/caption/")  # 0.0.0.0:1234/caption/
async def run_captioning(req: Request):
    try:
        model_output = generate(req.inputs, temperature=req.temperature)
    except Exception as e:
        return {"message": e}
    return {"output": model_output}


def caption(inputs: List[str], temperature: float = 0.01, url: str = None):
    """
    inputs should be a list of text prompts / image urls in arbitrary order.

    The urls will be embedded as an image, and any other arbitrary string as text.
    """
    URL = url or f"http://{HOST}:{PORT}/"
    if not URL.endswith("/"):
        URL = URL + "/"
    resp = requests.get(
        f"{URL}caption/",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={"inputs": inputs, "temperature": temperature},
    )
    return resp.json().get("output", resp.json())


if __name__ == "__main__":
    # run uvicorn app
    import uvicorn

    uvicorn.run(
        "inference_api:app",
        host=HOST,
        port=int(PORT),
        log_level="info",
        workers=1,
    )