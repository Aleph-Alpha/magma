import torch
import torch.nn.functional as F
from rational.utils import find_weights

def swish(x):
    return x * torch.sigmoid(x)

act_names = ["relu", "leaky_relu", "tanh", "gelu", "sigmoid", "swish"]
act_func = [F.relu, F.leaky_relu, torch.tanh, F.gelu, torch.sigmoid, swish]
versions = ["A", "B", "C", "D"]

for act_n, act_f in zip(act_names, act_func):
    print("-" * 30)
    print(f"Computing weights for {act_n}")
    print("-" * 30)
    show_plot = False
    save_in_file = True
    overwrite = True
    for version in versions:
        find_weights(act_f, act_n, (5, 4), (-3, 3), version, show_plot,
                     save_in_file, overwrite)
