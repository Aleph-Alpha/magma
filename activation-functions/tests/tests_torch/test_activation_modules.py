from unicodedata import category
from rational.torch import activation_modules as af
import torch


inp = torch.stack([(torch.rand(10000)-(i+1))*2 for i in range(5)], 1)

device="cpu"
test_lrelu = af.LReLu(device)
test_lrelu.input_retrieve_mode(mode="categories", category_name="neg")
test_lrelu(inp)
test_lrelu.show()

