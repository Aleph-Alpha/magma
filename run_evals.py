import os
import traceback

paths = {
    "nfresnet_no_adapters": "/mnt/shared_vol/checkpoints/multimodal_transformer_nfresnet_no_adapters_frozen_baseline_wd_0",
    "nfresnet_adapters": "/mnt/shared_vol/checkpoints/multimodal_transformer_nfresnet_adapters_baseline",
    "RN50x4_base": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x4_base",
    "RN50x16_base": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_base_2",
    "RN50x16_attn_adapter": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_base_adapter_post_attn",
    "RN50x16_attn_adapter_prioritize_ff": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_base_adapter_post_attn_attn_downsample_6_ff_downsample_12",
    "RN50x16_scaled_parallel_adapter": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_scaled_parallel_adapter_wd_fix",
    "RN50x16_base_large": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_base_large",  # larger downsample dim
    "ViT_base": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_clip_vit",
    "blind": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_blind_3",
    "RN50x16_adapter_post_attn_scaled_parallel": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_adapter_attn_scaled_parallel",
}

configs = {
    "nfresnet_no_adapters": "ablations/frozen_baseline_nfresnet_seq2.yml",
    "nfresnet_adapters": "ablations/nfresnet_plus_adapters.yml",
    "RN50x4_base": "ablations/base_resnet_small.yml",
    "RN50x16_base": "ablations/base.yml",
    "RN50x16_attn_adapter": "ablations/base_adapter_post_attn.yml",
    "RN50x16_attn_adapter_prioritize_ff": "ablations/base_adapter_post_attn_ff_prioritize.yml",
    "RN50x16_scaled_parallel_adapter": "ablations/base_parallel_adapter.yml",
    "RN50x16_base_large": "ablations/base_large.yml",  # larger downsample dim
    "ViT_base": "ablations/base_clip_vit.yml",
    "blind": "ablations/base_blind_3.yml",
    "RN50x16_adapter_post_attn_scaled_parallel": "ablations/adapter_attn_scaled_parallel.yml",
}

for name, path in paths.items():
    checkpoint = os.path.join(path, "global_step15000/mp_rank_00_model_states.pt")
    if name == "nfresnet_no_adapters":
        checkpoint = os.path.join(
            path, "global_step30000/mp_rank_00_model_states.pt"
        )  # trained at a smaller batch size

    config_path = configs[name]
    cmd = f"deepspeed eval.py {config_path} --checkpoint {checkpoint} --data_dir /mnt/localdisk/ \
        --tasks vqa gqa okvqa vizwiz coco --few_shot_examples 0 1 4 8 --max_n_steps 2000 --save_path eval_results/{name}.json"
    try:
        os.system(cmd)
    except:
        print(f"Failed to run {name}")
        exc = traceback.format_exc()
        # save exc to file
        with open(f"{name}_error.txt", "w") as f:
            f.write(exc)
