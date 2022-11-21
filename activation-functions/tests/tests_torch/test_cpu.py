import torch
from torch.nn.functional import leaky_relu
from rational.torch import Rational
import numpy as np


t = torch.tensor([-2., -1, 0., 1., 2.])
expected_res = np.array(leaky_relu(t))
inp = torch.from_numpy(np.array(t)).reshape(-1)
cuda_inp = torch.tensor(np.array(t), dtype=torch.float,
                        device="cuda").reshape(-1)


rationalA_lrelu_cpu = Rational(version='A', cuda=False)(inp).detach().numpy()
rationalB_lrelu_cpu = Rational(version='B', cuda=False)(inp).detach().numpy()
rationalC_lrelu_cpu = Rational(version='C', cuda=False)(inp).detach().numpy()
rationalD_lrelu_cpu = Rational(
    version='D', cuda=False, trainable=False)(inp).detach().numpy()

rationalA_lrelu_gpu = Rational(version='A', cuda=True)(
    cuda_inp).clone().detach().cpu().numpy()
rationalB_lrelu_gpu = Rational(version='B', cuda=True)(
    cuda_inp).clone().detach().cpu().numpy()
rationalC_lrelu_gpu = Rational(version='C', cuda=True)(
    cuda_inp).clone().detach().cpu().numpy()
rationalD_lrelu_gpu = Rational(version='D', cuda=True, trainable=False)(
    cuda_inp).clone().detach().cpu().numpy()


#  Tests on cpu
def test_rationalA_cpu_lrelu():
    assert np.all(np.isclose(rationalA_lrelu_cpu, expected_res, atol=5e-02))


def test_rationalB_cpu_lrelu():
    assert np.all(np.isclose(rationalB_lrelu_cpu, expected_res, atol=5e-02))


def test_rationalC_cpu_lrelu():
    assert np.all(np.isclose(rationalC_lrelu_cpu, expected_res, atol=5e-02))


def test_rationalD_cpu_lrelu():
    assert np.all(np.isclose(rationalD_lrelu_cpu, expected_res, atol=5e-02))
    # print(rationalD_lrelu_cpu)
