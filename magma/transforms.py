from torchvision import transforms as T
import torch.nn.functional as F
from PIL import ImageOps
import PIL
import random


def pad_to_size(x, size=256):
    delta_w = size - x.size[0]
    delta_h = size - x.size[1]
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )
    new_im = ImageOps.expand(x, padding)
    return new_im


def pad_to_size_tensor(x, size=256):
    offset_dim_1 = size - x.shape[1]
    offset_dim_2 = size - x.shape[2]

    padding_dim_1 = max(offset_dim_1 // 2, 0)
    padding_dim_2 = max(offset_dim_2 // 2, 0)

    if offset_dim_1 % 2 == 0:
        pad_tuple_1 = (padding_dim_1, padding_dim_1)
    else:
        pad_tuple_1 = (padding_dim_1 + 1, padding_dim_1)

    if offset_dim_2 % 2 == 0:
        pad_tuple_2 = (padding_dim_2, padding_dim_2)
    else:
        pad_tuple_2 = (padding_dim_2 + 1, padding_dim_2)

    padded = F.pad(x, pad=(*pad_tuple_2, *pad_tuple_1, 0, 0))
    return padded


class RandCropResize(object):

    """
    Randomly crops, then randomly resizes, then randomly crops again, an image. Mirroring the augmentations from https://arxiv.org/abs/2102.12092
    """

    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        img = pad_to_size(img, self.target_size)
        d_min = min(img.size)
        img = T.RandomCrop(size=d_min)(img)
        t_min = min(d_min, round(9 / 8 * self.target_size))
        t_max = min(d_min, round(12 / 8 * self.target_size))
        t = random.randint(t_min, t_max + 1)
        img = T.Resize(t)(img)
        if min(img.size) < 256:
            img = T.Resize(256)(img)
        return T.RandomCrop(size=self.target_size)(img)


def get_transforms(
    image_size, encoder_name, input_resolution=None, use_extra_transforms=False
):
    if "clip" in encoder_name:
        assert input_resolution is not None
        return clip_preprocess(input_resolution)

    base_transforms = [
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        RandCropResize(image_size),
        T.RandomHorizontalFlip(p=0.5),
    ]
    if use_extra_transforms:
        extra_transforms = [T.ColorJitter(0.1, 0.1, 0.1, 0.05)]
        base_transforms += extra_transforms
    base_transforms += [
        T.ToTensor(),
        maybe_add_batch_dim,
    ]
    base_transforms = T.Compose(base_transforms)
    return base_transforms


def maybe_add_batch_dim(t):
    if t.ndim == 3:
        return t.unsqueeze(0)
    else:
        return t


def pad_img(desired_size):
    def fn(im):
        old_size = im.size  # old_size[0] is in (width, height) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        im = im.resize(new_size, PIL.Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = PIL.Image.new("RGB", (desired_size, desired_size))
        new_im.paste(
            im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2)
        )

        return new_im

    return fn


def crop_or_pad(n_px, pad=False):
    if pad:
        return pad_img(n_px)
    else:
        return T.CenterCrop(n_px)


def clip_preprocess(n_px, use_pad=False):
    return T.Compose(
        [
            T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
            crop_or_pad(n_px, pad=use_pad),
            lambda image: image.convert("RGB"),
            T.ToTensor(),
            maybe_add_batch_dim,
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
