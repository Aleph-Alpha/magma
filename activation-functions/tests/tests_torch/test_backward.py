import torch
from torch.nn import MSELoss
from rational.torch import Rational
import numpy as np
import seaborn as sns
sns.set_theme()

visu_epochs = [0, 2, 4, 7, 10, 50, 100, 200]

def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))

def backward_test(cuda, version, recurrent_rat):
    inp = torch.arange(-4., 4., 0.1)
    if cuda:
        inp = inp.cuda()
    exp = torch.sigmoid(inp)
    rat = Rational(cuda=cuda)
    if recurrent_rat:
        def rat_func(inp):
            return rat(rat(inp))
    else:
        rat_func = rat
    loss_fn = MSELoss()
    optimizer = torch.optim.SGD(rat.parameters(), lr=0.01, momentum=0.9)
    # rat.input_retrieve_mode()
    for i in range(1000):
        out = rat_func(inp)
        optimizer.zero_grad()
        loss = loss_fn(out, exp)
        loss.backward()
        optimizer.step()
        if not i % 200:
            import matplotlib.pyplot as plt
            plt.plot(inp.detach().numpy(), rat_func(inp).detach().numpy())
            plt.plot(inp.detach().numpy(), sigmoid_np(inp.detach().numpy()))
            plt.legend("recrat")
            plt.show()


# def backward_test_hist(cuda, version, recurrent_rat):
#     r1, r2 = -3., 3.
#     rat = Rational(cuda=cuda)
#     rat.input_retrieve_mode()
#     if recurrent_rat:
#         def rat_func(inp):
#             return rat(rat(inp))
#     else:
#         rat_func = rat
#     loss_fn = MSELoss()
#     optimizer = torch.optim.SGD(rat.parameters(), lr=0.05, momentum=0.9)
#     for i in range(201):
#         inp = (r1 - r2) * torch.rand(400) + r2
#         if cuda:
#             inp = inp.cuda()
#         exp = torch.sigmoid(inp)
#         out = rat_func(inp)
#         optimizer.zero_grad()
#         loss = loss_fn(out, exp)
#         loss.backward()
#         optimizer.step()
#         if i in visu_epochs:
#             rat.capture(f"Epoch {i}")
#     rat.export_evolution_graph(f"snapplot.svg")
# 
#     # for snap in rat.snapshot_list:
#     #     snap.show(other_func=sigmoid_np)
#     import time
#     now = time.time()
#     rat.save_animated_graph(other_func=sigmoid_np)
#     print(time.time() - now)

# for cuda in [True, False]:
#     for version in ["A", "B", "C", "D"]:
#         for recurrence in [False, True]:
#             backward_test(cuda, version, recurrence)

# backward_test(False, "A", True)
# backward_test_hist(False, "A", False)
