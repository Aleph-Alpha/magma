import os
import traceback

paths = {
    "MAGMA_RN50x4_ff_prioritize": "/mnt/shared_vol/checkpoints/MAGMA_RN50x4_ff_prioritize",
    "MAGMA_RN50x16": "/mnt/shared_vol/checkpoints/MAGMA_RN50x16",
}

configs = {
    "MAGMA_RN50x4_ff_prioritize": "MAGMA/MAGMA_RN50x4_ff_prioritize.yml",
    "MAGMA_RN50x16": "MAGMA/MAGMA_v1.yml",
}

ckpt_steps = {"MAGMA_RN50x4_ff_prioritize": [30000], "MAGMA_RN50x16": [60000]}

task_dict = {
    "MAGMA_RN50x4_ff_prioritize": ["coco", "nocaps"],
    "MAGMA_RN50x16": ["coco", "nocaps"],
}

s = " "

for name, path in paths.items():
    for step in ckpt_steps[name]:
        checkpoint = os.path.join(path, f"global_step{step}/mp_rank_00_model_states.pt")
        config_path = configs[name]
        cmd = f'deepspeed --include localhost:2 eval.py {config_path} --checkpoint {checkpoint} --task_induction "A picture of" --data_dir /mnt/localdisk/ --tasks {s.join(task_dict[name])} --few_shot_examples  0 --save_path eval_results/MAGMA/{name}_step{step}_captioning.json'
        try:
            os.system(cmd)
        except:
            print(f"Failed to run {name}")
            exc = traceback.format_exc()
            # save exc to file
            with open(f"{name}_error.txt", "w") as f:
                f.write(exc)