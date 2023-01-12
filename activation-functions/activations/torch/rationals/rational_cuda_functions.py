import torch

if torch.cuda.is_available():
    # try:
    from activations.torch.rationals.cuda import *

    class Rational_CUDA_A_F(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input, weight_numerator, weight_denominator, training):

            ctx.save_for_backward(
                input, weight_numerator, weight_denominator)
            x = forward_A_5_4(input, weight_numerator, weight_denominator)

            return x

        @staticmethod
        def backward(ctx, grad_output):

            x, w_numerator, w_denominator = ctx.saved_tensors
            d_x, d_weight_numerator, d_weight_denominator = backward_A_5_4(
                grad_output.contiguous(), x, w_numerator, w_denominator)

            return d_x, d_weight_numerator, d_weight_denominator, None

    class Rational_CUDA_B_F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight_numerator, weight_denominator, training):
            ctx.save_for_backward(
                input, weight_numerator, weight_denominator)
            x = forward_B_5_4(input, weight_numerator, weight_denominator)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            x, w_numerator, w_denominator = ctx.saved_tensors
            d_x, d_weight_numerator, d_weight_denominator = backward_B_5_4(
                grad_output, x, w_numerator, w_denominator)
            return d_x, d_weight_numerator, d_weight_denominator, None

    class Rational_CUDA_C_F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight_numerator, weight_denominator, training):
            ctx.save_for_backward(
                input, weight_numerator, weight_denominator)
            x = forward_C_5_4(input, weight_numerator, weight_denominator)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            x, w_numerator, w_denominator = ctx.saved_tensors
            d_x, d_weight_numerator, d_weight_denominator = backward_C_5_4(
                grad_output, x, w_numerator, w_denominator)
            return d_x, d_weight_numerator, d_weight_denominator, None

    class Rational_CUDA_D_F(torch.autograd.Function):
        cnt = 0

        @staticmethod
        def forward(ctx, input, w_numerator, w_denominator, training):
            local_cnt = Rational_CUDA_D_F.cnt

            ctx.save_for_backward(input, w_numerator, w_denominator, torch.tensor(
                local_cnt, dtype=torch.long))

            Rational_CUDA_D_F.cnt += 1
            x = forward_D_5_4(training, local_cnt, input,
                              w_numerator, w_denominator)
            return x

        @staticmethod
        def backward(ctx, grad_output):
            # if not grad_output.is_contiguous():  # TODO this check is necessary if efficientnet is used
            #    grad_output = grad_output.contiguous()

            x, weight_numerator, weight_denominator, local_cnt = ctx.saved_tensors
            d_x, d_weight_numerator, d_weight_denominator = backward_D_5_4(True,
                                                                           local_cnt,
                                                                           grad_output,
                                                                           x,
                                                                           weight_numerator,
                                                                           weight_denominator)

            return d_x, d_weight_numerator, d_weight_denominator, None

    # except ImportError as Err:
    #     from warnings import warn
    #     msg = "You haven't installed the CUDA optimized pytorch version of " \
    #           "the rational-activations package"
    #     warn(msg)

    #     def _get_xps(z, len_numerator, len_denominator):
    #         xps = list()
    #         xps.append(z)
    #         for _ in range(max(len_numerator, len_denominator) - 2):
    #             xps.append(xps[-1].mul(z))
    #         xps.insert(0, torch.ones_like(z))
    #         return torch.stack(xps, 1)

    #     def Rational_CUDA_A_F(x, weight_numerator, weight_denominator, training):
    #         # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #         #               1 + | b_1 * X | + | b_2 * X^2| + ... + | b_m * X ^m|
    #         device = weight_numerator.device
    #         z = x.view(-1)
    #         len_num, len_deno = len(weight_numerator), len(weight_denominator)
    #         # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    #         xps = _get_xps(z, len_num, len_deno)
    #         numerator = xps.mul(weight_numerator).sum(1)
    #         expanded_dw = torch.cat([torch.tensor([1.]).to(device),
    #                                  weight_denominator,
    #                                  torch.zeros(len_num -
    #                                              len_deno - 1).to(device)])
    #         denominator = xps.mul(expanded_dw).abs().sum(1)
    #         return numerator.div(denominator).view(x.shape)

    #     def Rational_CUDA_B_F(x, weight_numerator, weight_denominator, training):
    #         # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #         #               1 + |b_1 * X + b_1 * X^2 + ... + b_m * X^m|
    #         z = x.view(-1)
    #         len_num, len_deno = len(weight_numerator), len(weight_denominator)
    #         # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    #         xps = _get_xps(z, len_num, len_deno)
    #         numerator = xps.mul(weight_numerator).sum(1)
    #         denominator = xps[:, 1:len_deno +
    #                           1].mul(weight_denominator).sum(1).abs()
    #         return numerator.div(1 + denominator).view(x.shape)

    #     def Rational_CUDA_C_F(x, weight_numerator, weight_denominator, training):
    #         # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #         #               eps + |b_0 + b1 * X + b_2 * X^2 + ... + b_m*X^m|
    #         z = x.view(-1)
    #         len_num, len_deno = len(weight_numerator), len(weight_denominator)
    #         # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    #         xps = _get_xps(z, len_num, len_deno)
    #         numerator = xps.mul(weight_numerator).sum(1)
    #         denominator = xps[:, :len_deno].mul(
    #             weight_denominator).sum(1).abs()
    #         return numerator.div(0.1 + denominator).view(x.shape)

    #     def Rational_CUDA_D_F(x, weight_numerator, weight_denominator, training, random_deviation=0.1):
    #         # P(X)/Q(X) = noised(a_0) + noised(a_1) * X +noised(a_2) * X^2 + ... + noised(a_n) * X^n /
    #         #     #                1 + |noised(b_1) * X + noised(b_2) * X^2 + ... + noised(b_m)*X^m|
    #         #     # Noised parameters have uniform noise to be in range [(1-random_deviation)*parameter,(1+random_deviation)*parameter].
    #         if not training:
    #             # do not add noise
    #             return Rational_CUDA_B_F(x, weight_numerator, weight_denominator, training)
    #         z = x.view(-1)
    #         len_num, len_deno = len(weight_numerator), len(weight_denominator)
    #         # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    #         xps = _get_xps(z, len_num, len_deno)
    #         numerator = xps.mul(weight_numerator.mul(
    #             torch.FloatTensor(len_num).uniform_(1-random_deviation,
    #                                                 1+random_deviation))
    #                             ).sum(1)
    #         denominator = xps[:, 1:len_deno +
    #                           1].mul(weight_denominator).sum(1).abs()
    #         return numerator.div(1 + denominator).view(x.shape)

    #     def Rational_NONSAFE_F(x, weight_numerator, weight_denominator, training):
    #         # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #         #               1 + b_1 * X + b_1 * X^2 + ... + b_m * X^m
    #         z = x.view(-1)
    #         len_num, len_deno = len(weight_numerator), len(weight_denominator)
    #         # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    #         xps = _get_xps(z, len_num, len_deno)
    #         numerator = xps.mul(weight_numerator).sum(1)
    #         denominator = xps[:, 1:len_deno+1].mul(weight_denominator).sum(1)
    #         return numerator.div(1 + denominator).view(x.shape)
