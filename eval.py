# %%

from magma.magma import (
    Magma,
)
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['MASTER_ADDR'] = 'localhost'
# modify if RuntimeError: Address already in use
os.environ['MASTER_PORT'] = '9994'
os.environ['RANK'] = "0"
os.environ['LOCAL_RANK'] = "0"
os.environ['WORLD_SIZE'] = "1"

model = Magma('/home/ml-mmeuer/adaptable_magma/fb20-dgx2-configs/Flamingo.yml')

print(list(model.named_children()))

# %%
