import torch
import os
import deepspeed
import wandb
from torch.utils.data import random_split, ConcatDataset
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
from functools import partial
from torchvision.utils import make_grid
from multimodal_fewshot.datasets import (
    MultimodalDataset,
    collate_fn,
    get_dataset,
    ImgCptDataset,
    VQADataset,
    VQAFewShot,
    GQADataset,
    GQAFewShot,
    vqa_eval_step,
    gqa_eval_step,
    GQAFewShotNew,
    VQAFewShotNew,
)
from multimodal_fewshot.datasets.vqa_eval import GQAFewShotNew, VQAFewShotNew
from multimodal_fewshot.datasets.snli_ve import SNLI_VE_Dataset
from multimodal_fewshot.datasets.nlvr2 import NLVR2Dataset
from multimodal_fewshot.datasets.vizwiz import VizWizDataset, VizWizFewShot
from multimodal_fewshot.datasets.dataset import ClassificationWrapper
from multimodal_fewshot.transforms import get_transforms
from multimodal_fewshot.model import (
    MultimodalLM,
    MultimodalClassifier,
    get_language_model,
)
from multimodal_fewshot.utils import (
    count_parameters,
    is_main,
    cycle,
    get_tokenizer,
    parse_args,
    wandb_log,
    wandb_init,
    save_model,
    load_model,
    print_main,
    configure_param_groups,
    log_table,
    collate_fn_classification,
)
from multimodal_fewshot.config import MultimodalConfig
from multimodal_fewshot.train_loop import (
    eval_step,
    eval_step_classification,
    inference_step,
    train_step,
    train_step_classification,
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
    dataset_type = getattr(config, "dataset_type", "old")
    if dataset_type == "new":
        # if config.train_dataset_dir is a list, load all datasets + join together
        train_dataset = _load_img_cpt_datasets(
            config.train_dataset_dir, tokenizer, transforms
        )
        if config.eval_dataset_dir is None:
            eval_len = 60000
            train_len = len(train_dataset) - eval_len
            print(
                f"Randomly splitting train_dataset into two datasets of length {train_len} and {eval_len}"
            )
            train_dataset, eval_dataset = random_split(
                train_dataset, [train_len, eval_len]
            )
        else:
            eval_dataset = _load_img_cpt_datasets(
                config.eval_dataset_dir, tokenizer, transforms
            )
        fewshot_vqa = cycle(
            iter(VQAFewShotNew(ImgCptDataset(config.vqa_dir, tokenizer, transforms)))
        )
        fewshot_gqa = cycle(
            iter(GQAFewShotNew(ImgCptDataset(config.gqa_dir, tokenizer, transforms)))
        )

    elif dataset_type == "vizwiz":

        train_dataset = VizWizDataset(
            data_dir=config.train_dataset_dir,
            tokenizer=tokenizer,
            transforms=transforms,
            mode="train",
            load_images=True,
            return_img_cpt=True,
        )
        eval_dataset = VizWizDataset(
            data_dir=config.eval_dataset_dir,
            tokenizer=tokenizer,
            transforms=transforms,
            mode="val",
            load_images=True,
            return_img_cpt=True,
        )

        fewshot_vqa = cycle(
            iter(
                VizWizFewShot(
                    data_dir=config.eval_dataset_dir,
                    tokenizer=tokenizer,
                    transforms=transforms,
                )
            )
        )

        fewshot_gqa = cycle(
            iter(GQAFewShotNew(ImgCptDataset(config.gqa_dir, tokenizer, transforms)))
        )

    elif dataset_type == "old":

        dataset = MultimodalDataset(
            get_dataset(config.train_dataset_name, data_dir=config.train_dataset_dir),
            seq_len=config.seq_len,
            tokenizer=tokenizer,
            transforms=transforms,
        )
        train_dataset, eval_ds1 = random_split(dataset, [len(dataset) - 60000, 60000])
        eval_ds2 = MultimodalDataset(
            get_dataset(config.eval_dataset_name, data_dir=config.eval_dataset_dir),
            seq_len=config.seq_len,
            tokenizer=tokenizer,
            transforms=transforms,
        )
        eval_dataset = ConcatDataset([eval_ds1, eval_ds2])

        fewshot_vqa = cycle(
            iter(
                VQAFewShot(VQADataset(config.vqa_dir), tokenizer, transforms=transforms)
            )
        )

        fewshot_gqa = cycle(
            iter(
                GQAFewShot(
                    data_dir=config.gqa_dir,
                    mode="val",
                    tokenizer=tokenizer,
                    transforms=transforms,
                )
            )
        )

    print_main(f"Loaded train dataset with {len(train_dataset)} samples")
    print_main(f"Loaded eval dataset with {len(eval_dataset)} samples")

    return train_dataset, eval_dataset, fewshot_vqa, fewshot_gqa


def get_classification_datasets(config, tokenizer, transforms):
    dataset_type = getattr(config, "dataset_type", "old")
    if dataset_type == "new":
        train_dataset = ImgCptDataset(
            config.train_dataset_dir, tokenizer=tokenizer, transforms=transforms
        )
        eval_dataset = ImgCptDataset(
            config.eval_dataset_dir, tokenizer=tokenizer, transforms=transforms
        )
        train_dataset = ClassificationWrapper(
            train_dataset, config.class_dict["num_classes"]
        )
        eval_dataset = ClassificationWrapper(
            eval_dataset, config.class_dict["num_classes"]
        )
    elif dataset_type == "snli":
        train_dataset = SNLI_VE_Dataset(
            config.train_dataset_dir, tokenizer, transforms, mode="train"
        )
        eval_dataset = SNLI_VE_Dataset(
            config.eval_dataset_dir, tokenizer, transforms, mode="val"
        )
    elif dataset_type == "nlvr2":
        train_dataset = NLVR2Dataset(
            config.train_dataset_dir,
            "train",
            tokenizer=tokenizer,
            transforms=transforms,
        )
        eval_dataset = NLVR2Dataset(
            config.eval_dataset_dir, "val", tokenizer=tokenizer, transforms=transforms
        )
    else:
        raise ValueError(f"Dataset type {dataset_type} not recognized")

    return train_dataset, eval_dataset


# tell tokenizers not to do parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":

    # parse command line arguments:
    args = parse_args()
    deepspeed.init_distributed()

    config = MultimodalConfig.from_yml(args.config)
    config.print()

    # load model + tokenizer:
    tokenizer = get_tokenizer(config.tokenizer_name)
    if config.is_classifier:
        model = MultimodalClassifier.from_pretrained(
            config
        )  # we might not always want to do this if e.g we're continuing a classification training? Although I guess those weights get loaded later?
    else:
        model = MultimodalLM(
            lm=get_language_model(config.lm_name, model_dir="/mnt/localdisk/models/"),
            tokenizer=tokenizer,
            config=config,
        )

    if config.seq_len is None:
        config.seq_len = model.lm.config.max_position_embeddings
    if is_main():
        n_params = count_parameters(model)
        print(f"Training model with {n_params:,} trainable parameters")
        config.num_trainable_parameters = n_params
    transforms = get_transforms(config.image_size, model,)
    trainable_parameters = configure_param_groups(model, config)

    # load data:
    if not config.is_classifier:
        # then we are in pretraining, return pretraining datasets
        (
            train_dataset,
            eval_dataset,
            fewshot_vqa,
            fewshot_gqa,
        ) = get_pretraining_datasets(config, tokenizer, transforms)
    else:
        # then we are in classification / fine tuning, return classification datasets
        train_dataset, eval_dataset = get_classification_datasets(
            config, tokenizer, transforms
        )
        assert (
            train_dataset.num_classes == eval_dataset.num_classes
        ), "num classes mismatch"
        assert (
            train_dataset.num_classes == config.class_dict["num_classes"]
        ), f"The number of classes in the train dataset ({train_dataset.num_classes}) must match the number of classes in the config ({config.class_dict['num_classes']})"

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
        collate_fn=partial(collate_fn, seq_len=config.seq_len)
        if not config.is_classifier
        else partial(collate_fn_classification, seq_len=config.seq_len),
        config_params=config.deepspeed_config_params,
    )
    eval_loader = cycle(model_engine.deepspeed_io(eval_dataset))
    train_loader = cycle(train_loader)

    # start train loop
    global_step = 0
    if config.load:
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

    for i in pbar:
        if global_step >= config.train_steps:
            break

        ##### train step
        if config.is_classifier:
            loss, acc = train_step_classification(config, train_loader, model_engine)
            wandb_log({"train/acc": acc}, step=global_step)
        else:
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
                if config.is_classifier:
                    eval_loss, eval_acc = eval_step_classification(
                        config, train_loader, model_engine
                    )
                    wandb_log({"eval/acc": eval_acc}, step=global_step)
                else:
                    eval_loss = eval_step(config, eval_loader, model_engine)

                wandb_log({"eval/loss": eval_loss}, step=global_step)
                pbar.set_description(
                    f"evaluating... Step: {global_step} Eval Loss: {eval_loss}"
                )

                if not config.is_classifier:
                    ##### inference:
                    image_grid, caption = inference_step(
                        config, eval_loader, model_engine
                    )
                    wandb_log(
                        {"inference/image": wandb.Image(image_grid, caption=caption)},
                        step=global_step,
                    )

                    ##### vqa eval:
                    vqa_acc, model_outputs, gt_answers_list = vqa_eval_step(
                        config, model_engine, fewshot_vqa
                    )
                    wandb_log({"eval/vqa_acc": vqa_acc}, step=global_step)
                    log_table(
                        "vqa_results", model_outputs, gt_answers_list, global_step
                    )

                    ##### gqa eval
                    gqa_acc, model_outputs, gt_answers_list = gqa_eval_step(
                        config, model_engine, fewshot_gqa
                    )
                    wandb_log({"eval/gqa_acc": gqa_acc}, step=global_step)
                    log_table(
                        "gqa_results", model_outputs, gt_answers_list, global_step
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
