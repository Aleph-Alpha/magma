from rational.torch import Rational

rat_l = Rational("leaky_relu")
rat_s = Rational("sigmoid")
rat_i = Rational("identity")

rat_l.show()

print(Rational.list)
# [Rational Activation Function A) of degrees (5, 4) running on cuda 0x7f778678b700
# , Rational Activation Function A) of degrees (5, 4) running on cuda 0x7f778678b1c0
# , Rational Activation Function A) of degrees (5, 4) running on cuda 0x7f77851fb5b0
# ]

Rational.show_all()

print(rat_l.snapshot_list)
rat_l.capture(name="Leaky init :)")
print(rat_l.snapshot_list)
# []
# [Snapshot (Leaky init :))]

import torch
rat_l.snapshot_list[0].show(other_func=[torch.sin, torch.tanh])

Rational.show_all(other_func=[torch.sin, torch.tanh])


import matplotlib.pyplot as plt
import seaborn as sns

with sns.axes_style("whitegrid"):
    ax = plt.gca()

rat_i.func_name = "new_name"

for rat in Rational.list:
    rat.show(title="Different initialisations", axis=ax)
plt.legend()
plt.show()

Rational.show_all(title="Different initialisations", axes=ax)  # equivalent
