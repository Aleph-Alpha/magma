import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from .utils import reduce_losses, to_cuda_half
from torchvision.utils import make_grid
import argparse
import os
local_rank = int(os.getenv('LOCAL_RANK', None))
DEVICE = torch.device(
    f"cuda:{local_rank}" if local_rank != None else "cpu"
)


def train_step(config, train_loader, model_engine, scaler):
    losses = []
    # with torch.autograd.set_detect_anomaly(True):
    # with torch.cuda.amp.autocast():
    for _ in range(config.gradient_accumulation_steps):
        images, captions = next(train_loader)
        images, captions = images.half().to(DEVICE), captions.to(DEVICE)
        if config.run_blind:
            images = torch.zeros_like(images)

        outputs = model_engine(images, captions)
        loss = outputs.loss

        losses.append(loss)
        model_engine.backward(loss)
        model_engine.step()

    return reduce_losses(torch.mean(torch.stack(losses))).item()


def train_step_classification(config, train_loader, model_engine, return_accuracy=True):
    losses = []
    if return_accuracy:
        accuracies = []
    for _ in range(config.gradient_accumulation_steps):
        images, captions, class_labels = next(train_loader)
        images, captions, class_labels = to_cuda_half(
            images, captions, class_labels)
        if config.run_blind:
            images = torch.zeros_like(images)
        loss, logits = model_engine(images, captions, class_labels)
        losses.append(loss)
        if return_accuracy:
            argmax_pred = logits.argmax(dim=-1)
            accuracies.append((argmax_pred == class_labels).float().mean())
        model_engine.backward(loss)
        model_engine.step()

    loss_reduced = reduce_losses(torch.mean(torch.stack(losses))).item()
    if return_accuracy:
        accuracy_reduced = reduce_losses(
            torch.mean(torch.stack(accuracies))).item()
        return loss_reduced, accuracy_reduced
    return loss_reduced


def eval_step(config, eval_loader, model_engine, device):
    losses = []

    for i in tqdm(range(config.eval_steps), "evaluating..."):
        images, captions = next(eval_loader)
        images, captions = images.half().to(device), captions.to(device)
        if config.run_blind:
            images = torch.zeros_like(images)
        outputs = model_engine(images, captions)
        loss = outputs.loss
        losses.append(loss)

    return reduce_losses(torch.mean(torch.stack(losses))).item()


def eval_step_classification(config, train_loader, model_engine, return_accuracy=True):
    losses = []
    if return_accuracy:
        accuracies = []

    for _ in range(config.gradient_accumulation_steps):
        images, captions, class_labels = next(train_loader)
        images, captions, class_labels = to_cuda_half(
            images, captions, class_labels)
        if config.run_blind:
            images = torch.zeros_like(images)

        loss, logits = model_engine(images, captions, class_labels)
        losses.append(loss)
        if return_accuracy:
            argmax_pred = logits.argmax(dim=-1)
            accuracies.append((argmax_pred == class_labels).float().mean())

    loss_reduced = reduce_losses(torch.mean(torch.stack(losses))).item()
    if return_accuracy:
        accuracy_reduced = reduce_losses(
            torch.mean(torch.stack(accuracies))).item()
        return loss_reduced, accuracy_reduced
    return loss_reduced


def inference_step(config, eval_loader, model_engine):
    images, _ = next(eval_loader)
    images = images.half().cuda()
    if config.run_blind:
        images = torch.zeros_like(images)
    captions = model_engine(
        images, captions=None, inference=True
    )  # [caption1, caption2, ... b]
    width = min(2, images.shape[0])
    image_grid = make_grid(images[:width])
    caption = ""
    for i in range(width):
        caption += f"Caption {i}: \n{captions[i]}\n"
    return image_grid, caption
