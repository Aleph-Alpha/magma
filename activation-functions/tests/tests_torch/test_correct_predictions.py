import torch
from torch.nn.functional import leaky_relu
from rational.torch import Rational
import numpy as np


t = torch.tensor([-2., -1, 0., 1., 2.])
expected_res = np.array(leaky_relu(t))
inp = torch.from_numpy(np.array(t)).reshape(-1)
cuda_inp = torch.tensor(np.array(t), dtype=torch.float, device="cuda").reshape(-1)

# todo: lrelu, tanh, sigmoid, ...)