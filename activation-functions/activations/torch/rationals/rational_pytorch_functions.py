import torch
from einops import rearrange


def _get_xps(z, len_numerator, len_denominator):
    xps = list()
    xps.append(z)
    for _ in range(max(len_numerator, len_denominator+1) - 2):
        xps.append(xps[-1].mul(z))
    xps.insert(0, torch.ones_like(z))
    return torch.stack(xps, 1)


def Rational_PYTORCH_A_F(x, weight_numerator, weight_denominator, device, pre, post):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #               1 + | b_1 * X | + | b_2 * X^2| + ... + | b_m * X ^m|

    len_num, len_deno = len(weight_numerator), len(weight_denominator)

    z = x.view(-1)
    # print("Before Vander", torch.equal(x, x_old))
    xps = torch.vander(z, N=max(len_num, len_deno), increasing=True)
    xps = torch.nan_to_num(xps, posinf=1/1024, neginf=-1/1024)

    # print(xps.isnan().any())
    # print('xps', torch.isinf(xps).any())

    # xps = _get_xps(z, len_num, len_deno)

    # print(f'{xps.device} {xps.dtype} {weight_numerator.device} {weight_numerator.dtype}')
    numerator_mul = xps.mul(weight_numerator)
    # print('num')
    # print(numerator_mul.isnan().any())
    # print(numerator_mul)
    # print('numerator_mul', torch.isinf(numerator_mul).any())
    # if torch.isinf(numerator_mul).any():

    numerator = numerator_mul.sum(-1)
    # numerator = torch.nan_to_num(numerator, posinf=fp16_max, neginf=-fp16_max)

    expanded_dw = torch.cat([pre, weight_denominator, post])
    denominator = xps.mul(expanded_dw)
    # print('denominator', torch.isinf(denominator).any())
    # print('denom')
    # print(denominator.isnan().any())
    denominator_ab = denominator.abs()
    denomi_sum = denominator_ab.sum(-1)
    # denomi_sum = torch.nan_to_num(
    #     denomi_sum, posinf=fp16_max, neginf=-fp16_max)

    out = numerator.div(denomi_sum)

    # print("OUt Vander x == x_old", torch.equal(x, x_old))

    new_out = out.view(x.shape)
    # new_out = rearrange(out, "(h1 h2 h3 h4) -> h1 h2 h3 h4",
    #                     h1=x.shape[0], h2=x.shape[1], h3=x.shape[2], h4=x.shape[3])
    # print('out')
    # print(new_out)
    return new_out


def Rational_PYTORCH_B_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #               1 + |b_1 * X + b_1 * X^2 + ... + b_m * X^m|
    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    xps = _get_xps(z, len_num, len_deno)
    numerator = xps.mul(weight_numerator).sum(1)
    denominator = xps[:, 1:len_deno+1].mul(weight_denominator).sum(1).abs()
    return numerator.div(1 + denominator).view(x.shape)


def Rational_PYTORCH_C_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #               eps + |b_0 + b1 * X + b_2 * X^2 + ... + b_m*X^m|
    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    xps = _get_xps(z, len_num, len_deno)
    numerator = xps.mul(weight_numerator).sum(1)
    denominator = xps[:, :len_deno].mul(weight_denominator).sum(1).abs()
    return numerator.div(0.1 + denominator).view(x.shape)


def Rational_PYTORCH_D_F(x, weight_numerator, weight_denominator, training, random_deviation=0.1):
    # P(X)/Q(X) = noised(a_0) + noised(a_1) * X +noised(a_2) * X^2 + ... + noised(a_n) * X^n /
    #     #                1 + |noised(b_1) * X + noised(b_2) * X^2 + ... + noised(b_m)*X^m|
    #     # Noised parameters have uniform noise to be in range [(1-random_deviation)*parameter,(1+random_deviation)*parameter].
    if not training:
        # do not add noise
        return Rational_PYTORCH_B_F(x, weight_numerator, weight_denominator, training)
    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    xps = _get_xps(z, len_num, len_deno)
    numerator = xps.mul(weight_numerator.mul(
        torch.FloatTensor(len_num).uniform_(1-random_deviation,
                                            1+random_deviation))
                        ).sum(1)
    denominator = xps[:, 1:len_deno+1].mul(weight_denominator).sum(1).abs()
    return numerator.div(1 + denominator).view(x.shape)


def Rational_NONSAFE_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #               1 + b_1 * X + b_1 * X^2 + ... + b_m * X^m
    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    xps = _get_xps(z, len_num, len_deno).to(weight_numerator.device)
    numerator = xps.mul(weight_numerator).sum(1)
    denominator = xps[:, 1:len_deno+1].mul(weight_denominator).sum(1)
    return numerator.div(1 + denominator).view(x.shape)


# def Rational_Spline_F(x, weight_numerator, weight_denominator, training):
#     # P(X) / Q(X) = (X - ~a_0) * (X + ~a0) * (a0 + a1*x + ... + a_n-1 * X^n-2) /
#     #               1 + b_1 * X + b_1 * X^2 + ... + b_m * X^m
#     k = weight_numerator[0]
#     z = x.view(-1)
#     len_num, len_deno = len(weight_numerator), len(weight_denominator)
#     xps = _get_xps(z, len_num-2, len_deno).to(weight_numerator.device)
#     numerator = (xps[:, :len_num-1].mul(weight_numerator[1:]).sum(1)).mul(torch.relu(z+k)).mul(-torch.relu(-z+k))
#     denominator = xps[:, 1:len_deno+1].mul(weight_denominator).sum(1).abs()
#     return numerator.div(1 + denominator).view(x.shape)

def Rational_Spline_F(x, k, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = (X - ~a_0) * (X + ~a0) * (a0 + a1*x + ... + a_n-1 * X^n-2) /
    #               1 + |b_1 * X + b_1 * X^2 + ... + b_m * X^m|
    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    xps = _get_xps(z, len_num, len_deno).to(weight_numerator.device)
    numerator = (xps[:, :len_num].mul(weight_numerator).sum(1)
                 ).mul(torch.relu(z+k)).mul(-torch.relu(-z+k))
    denominator = xps[:, 1:len_deno+1].mul(weight_denominator).sum(1).abs()
    return numerator.div(1 + denominator).view(x.shape)

# def Rational_Positive_Spline_F(x, k, weight_numerator, weight_denominator, training):
#     # P(X) / Q(X) = (X - ~a_0) * (X + ~a0) * (a0 + a1*x + ... + a_n-1 * X^n-2) /
#     #               1 + b_1 * X + b_1 * X^2 + ... + b_m * X^m
#     # k = weight_numerator[0]
#     z = x.view(-1)
#     len_num, len_deno = len(weight_numerator), len(weight_denominator)
#     xps = _get_xps(z, len_num-2, len_deno).to(weight_numerator.device)
#     numerator = (xps[:, :len_num-1].mul(weight_numerator[1:]).sum(1)).mul(torch.relu(z)).mul(-torch.relu(-z+k))
#     denominator = xps[:, 1:len_deno+1].mul(weight_denominator).sum(1).abs()
#     return numerator.div(1 + denominator).view(x.shape)


class Rational_CUDA_NONSAFE_F():
    def __init__(self):
        pass

    def apply():

        return Rational_NONSAFE_F
