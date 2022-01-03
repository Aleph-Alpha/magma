import os
import traceback

# paths = {
#     # "MAGMA_VQA": "/mnt/shared_vol/checkpoints/MAGMA_RN50x16_vqa",
#     # "MAGMA_MIX": "/mnt/shared_vol/checkpoints/MAGMA_RN50x16_mix",
#     # "MAGMA_OKVQA": "/mnt/shared_vol/checkpoints/MAGMA_RN50x16_okvqa",
#     # "MAGMA_NLVR": "/mnt/shared_vol/checkpoints/MAGMA_RN50x16_nlvr2",
#     # "MAGMA_SNLI": "/mnt/shared_vol/checkpoints/MAGMA_RN50x16_snli_ve",
#     # "MAGMA": "/mnt/shared_vol/checkpoints/MAGMA_RN50x16"
#     "MAGMA_COCO": "/mnt/shared_vol/checkpoints/MAGMA_RN50x16_coco"
# }

# configs = {
#     # "MAGMA_VQA": "finetunes/vqa.yml",
#     # "MAGMA_MIX": "finetunes/mix.yml",
#     # "MAGMA_OKVQA": "finetunes/okvqa.yml",
#     # "MAGMA_NLVR": "finetunes/nlvr2.yml",
#     # "MAGMA_SNLI": "finetunes/snli_ve.yml",
#     # "MAGMA": "MAGMA_v1.yml"
#     "MAGMA_COCO": "finetunes/coco.yml"
# }

# ckpt_steps = {
#     # "MAGMA_VQA": [500],
#     # "MAGMA_OKVQA": [25, 75, 125],
#     # "MAGMA_MIX": [2750],
#     # "MAGMA_NLVR": [250, 500, 750, 1000, 1250],
#     # "MAGMA_SNLI": [500, 1000, 1500],
#     # "MAGMA": [30000]
#     "MAGMA_COCO": [3500]
# }

paths = {
    # "nfresnet_no_adapters": "/mnt/shared_vol/checkpoints/multimodal_transformer_nfresnet_no_adapters_frozen_baseline_wd_0",
    # "nfresnet_adapters": "/mnt/shared_vol/checkpoints/multimodal_transformer_nfresnet_adapters_baseline",
    # "RN50x4_base": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x4_base",
    # "RN50x16_base": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_base_2",
    # "RN50x16_attn_adapter": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_base_adapter_post_attn",
    "RN50x16_attn_adapter_prioritize_ff": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_base_adapter_post_attn_attn_downsample_6_ff_downsample_12",
    # "RN50x16_scaled_parallel_adapter": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_scaled_parallel_adapter_wd_fix",
    # "RN50x16_base_large": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_base_large",  # larger downsample dim
    # "ViT_base": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_clip_vit",
    # "blind": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_blind_3",
    # "RN50x16_adapter_post_attn_scaled_parallel": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_adapter_attn_scaled_parallel",
    # "RN50x16_base_no_adapters": "/mnt/shared_vol/checkpoints/multimodal_transformer_rn50x16_no_adapters/",
}

configs = {
    # "nfresnet_no_adapters": "ablations/frozen_baseline_nfresnet_seq2.yml",
    # "nfresnet_adapters": "ablations/nfresnet_plus_adapters.yml",
    # "RN50x4_base": "ablations/base_resnet_small.yml",
    # "RN50x16_base": "ablations/base.yml",
    # "RN50x16_attn_adapter": "ablations/base_adapter_post_attn.yml",
    "RN50x16_attn_adapter_prioritize_ff": "ablations/base_adapter_post_attn_ff_prioritize.yml",
    # "RN50x16_scaled_parallel_adapter": "ablations/base_parallel_adapter.yml",
    # "RN50x16_base_large": "ablations/base_large.yml",  # larger downsample dim
    # "ViT_base": "ablations/base_clip_vit.yml",
    # "blind": "ablations/base_blind_3.yml",
    # "RN50x16_adapter_post_attn_scaled_parallel": "ablations/adapter_attn_scaled_parallel.yml",
    # "RN50x16_base_no_adapters": "ablations/base_no_adapters.yml",
}


for name, path in paths.items():
    checkpoint = os.path.join(path, "global_step15000/mp_rank_00_model_states.pt")
    if name == "nfresnet_no_adapters":
        checkpoint = os.path.join(
            path, "global_step30000/mp_rank_00_model_states.pt"
        )  # trained at a smaller batch size
    for temp in [0.01]:
        config_path = configs[name]
        cmd = f'deepspeed --include localhost:7 --master_port 6006 eval.py {config_path} --task_induction "A picture of" --checkpoint {checkpoint} --data_dir /mnt/localdisk/ --tasks nocaps --few_shot_examples  0 1 2 4 --save_path eval_results/{name}_temp_{temp}_nocaps_task_induction_step.json'
        # cmd = f'deepspeed --include localhost:4,5,6,7 --master_port 6969 eval.py {config_path} --task_induction "A picture of" --checkpoint {checkpoint} --data_dir /mnt/localdisk/ --tasks coco --few_shot_examples  0 --save_path eval_results/{name}_temp_{temp}_coco_task_induction_rerun.json'
        try:
            os.system(cmd)
        except:
            print(f"Failed to run {name}")
            exc = traceback.format_exc()
            # save exc to file
            with open(f"{name}_error.txt", "w") as f:
                f.write(exc)
