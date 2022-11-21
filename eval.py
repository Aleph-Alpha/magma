# %%

from magma.magma import (
    Magma,
)
import os
os.environ['MASTER_ADDR'] = 'localhost'
# modify if RuntimeError: Address already in use
os.environ['MASTER_PORT'] = '9994'
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"

print("LR",os.getenv('LOCAL_RANK'))

model = Magma.from_checkpoint('./configs/Switch_Rational_ReLU_Image_Encoder_From_Checkpoints.yml',
                              checkpoint_path='./model_checkpoints/multimodal_transformer_rn50x16/global_step150/zero_pp_rank_1_mp_rank_00_optim_states.pt')

print(list(model.named_children()))

# %%
