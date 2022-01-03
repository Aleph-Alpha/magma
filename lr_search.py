from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
import os
import itertools
import argparse


parser = argparse.ArgumentParser("wandb sweep argument parser")
parser.add_argument("--lr", type=float, help="learning rate")
parser.add_argument("--image_enc_lr", type=float, help="image encoder learning rate")
args = parser.parse_args()

experiment = {"lr": args.lr}

inp = "./configs/classification/sweep_base_snli.yml"
out = "./configs/classification/tmp.yml"


# make tmp yml with selected params changed
with open(inp, "r") as f:
    data = load(f, Loader=Loader)


data.update(experiment)

# delete any null values

to_delete = []
for k, v in data.items():
    if v is None:
        to_delete.append(k)

for k in to_delete:
    del data[k]

with open(out, "w") as f:
    f.write(dump(data))

cmd = "deepspeed train.py --config configs/classification/tmp.yml"
print(cmd)
try:
    os.system(cmd)
except BaseException as e:
    raise e
finally:
    # make sure run is killed
    os.system('pkill -f "deepspeed train.py"')
