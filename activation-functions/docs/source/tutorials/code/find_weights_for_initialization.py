from rational.utils.find_init_weights import find_weights
import torch.nn.functional as F  # To get the tanh function

find_weights(F.tanh)

# approximated function name: tanh
# approximated function name: tanh
# degree of the numerator P: 5
# degree of the denominator Q: 4
# lower bound: -3
# upper bound: 3
# Rational Version: B

# Found coeffient :
# P: [2.11729498e-09 9.99994250e-01 6.27633277e-07 1.07708645e-01
#  2.94655690e-08 8.71124374e-04]
# Q: [6.37690834e-07 4.41014181e-01 2.27476614e-07 1.45810399e-02]

# Do you want a plot of the result (y/n)y

# Do you want to store them in the json file ? (y/n)y

from rational.torch import Rational

rational_tanh_B = Rational("tanh", version="B")
print(rational_tanh_B.init_approximation)
# 'tanh'
print(rational_tanh_B.numerator.cpu().detach().numpy())
# [2.1172950e-09 9.9999428e-01 6.2763326e-07 1.0770865e-01 2.9465570e-08
#  8.7112439e-04]
print(rational_tanh_B.denominator.cpu().detach().numpy())
# [6.3769085e-07 4.4101417e-01 2.2747662e-07 1.4581040e-02]
