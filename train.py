import torch
import os
import deepspeed
import wandb
from torch.utils.data import random_split, ConcatDataset
from torch.optim import AdamW
from tqdm import tqdm
from functools import partial
from magma.datasets import (
    collate_fn,
    ImgCptDataset,
)
from magma.magma import (
    Magma,
)
from magma.utils import (
    is_main,
    cycle,
    parse_args,
    wandb_log,
    wandb_init,
    save_model,
    load_model,
    print_main,
    configure_param_groups,
)
from magma.train_loop import (
    eval_step,
    inference_step,
    train_step,
)


def _load_img_cpt_datasets(dataset_dir, tokenizer, transforms):
    if isinstance(dataset_dir, (list, tuple)):
        return ConcatDataset(
            [_load_img_cpt_datasets(d, tokenizer, transforms) for d in dataset_dir]
        )
    elif isinstance(dataset_dir, str):
        return ImgCptDataset(dataset_dir, tokenizer=tokenizer, transforms=transforms)
    else:
        raise TypeError("dataset dir wrong type")


def get_pretraining_datasets(config, tokenizer, transforms):
    # if config.train_dataset_dir is a list, load all datasets + join together
    train_dataset = _load_img_cpt_datasets(
        config.train_dataset_dir, tokenizer, transforms
    )
    # if no dedicated eval sets are given, use a percentage of the train dataset
    if config.eval_dataset_dir is None:
        eval_len = int(len(train_dataset) * config.eval_dataset_pct)
        train_len = len(train_dataset) - eval_len
        print(
            f"Randomly splitting train_dataset into two datasets of length {train_len} and {eval_len}"
        )
        train_dataset, eval_dataset = random_split(train_dataset, [train_len, eval_len])
    else:
        eval_dataset = _load_img_cpt_datasets(
            config.eval_dataset_dir, tokenizer, transforms
        )

    print_main(f"Loaded train dataset with {len(train_dataset)} samples")
    print_main(f"Loaded eval dataset with {len(eval_dataset)} samples")

    return train_dataset, eval_dataset


# tell tokenizers not to do parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":

    # parse command line arguments:
    args = parse_args()
    deepspeed.init_distributed()

    # load model + tokenizer:
    model = Magma(
        args.config
    )  # for finetuning one might want to load the model via Magma.from_checkpoint(...) here
    tokenizer, config, transforms = model.tokenizer, model.config, model.transforms

    # filter frozen from trainable parameters:
    trainable_parameters = configure_param_groups(model, config)

    # load data:
    train_dataset, eval_dataset = get_pretraining_datasets(
        config, tokenizer, transforms
    )

    print_main(f"Loaded train dataset with {len(train_dataset)} samples")
    print_main(f"Loaded eval dataset with {len(eval_dataset)} samples")

    opt = AdamW(
        trainable_parameters,
        config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )

    model_engine, opt, train_loader, lr_scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=opt,
        model_parameters=trainable_parameters,
        training_data=train_dataset,
        collate_fn=partial(collate_fn, seq_len=model.seq_len),
        config_params=config.deepspeed_config_params,
    )
    eval_loader = cycle(model_engine.deepspeed_io(eval_dataset))
    train_loader = cycle(train_loader)

    # initialize training
    global_step = 0
    if config.load:
        # loads a deepspeed checkpoint if provided. For finetuning, set load_optimizer to false
        previous_global_step = load_model(
            model_engine,
            config.load,
            load_optimizer_states=config.load_optimizer,
            load_lr_scheduler_states=config.load_optimizer,
        )

        if config.load_optimizer:
            global_step = previous_global_step

    pbar = tqdm(
        range(0, config.train_steps),
        desc="training...",
        initial=global_step,
        total=config.train_steps,
        disable=not is_main(),
    )
    wandb_init(
        project=config.wandb_project,
        name=config.name or wandb.util.generate_id(),
        config=config,
    )

    # training loop
    for i in pbar:
        if global_step >= config.train_steps:
            break

        ##### train step
        loss = train_step(config, train_loader, model_engine)

        global_step += 1

        if global_step % config.log_every == 0:
            pbar.set_description(f"training... Step: {global_step} Loss: {loss}")
            current_lr = (
                [lr for lr in lr_scheduler.get_lr()]
                if lr_scheduler is not None
                else config.lr
            )
            to_log = {"train/loss": loss, "train/lr": current_lr}
            wandb_log(to_log, step=global_step)

        ##### Evaluation phase
        if global_step % config.eval_every == 0:
            model_engine.eval()
            with torch.no_grad():

                ##### eval step:
                eval_loss = eval_step(config, eval_loader, model_engine)

                wandb_log({"eval/loss": eval_loss}, step=global_step)
                pbar.set_description(
                    f"evaluating... Step: {global_step} Eval Loss: {eval_loss}"
                )

                ##### inference:
                image_grid, caption = inference_step(config, eval_loader, model_engine)
                wandb_log(
                    {"inference/image": wandb.Image(image_grid, caption=caption)},
                    step=global_step,
                )

            model_engine.train()

        ##### Save model
        if global_step % config.save_every == 0:
            if config.save is not None:
                save_model(model_engine, config.save, global_step)
                print_main(f"saving model at step {global_step}")

    ##### Save model after training is finished
    if config.save is not None:
        save_model(model_engine, config.save, global_step)
        print_main(f"saving model at end of training (step {global_step})")
