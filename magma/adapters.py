import torch
import torch.nn as nn
from torchtyping import TensorType
from activations.torch import Rational
from activations.utils.convert_network import convert_pytorch_model_to_rational
from .utils import get_world_info
import os
import math


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):
        super().__init__()
        local_rank, rank, world_size = get_world_info()
        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":

            def swish(x):
                return x * torch.sigmoid(x)

            self.f = swish
        elif hidden_act.lower() == "gelu":

            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            self.f = gelu_new
        elif hidden_act.lower() == "gelu_orig":
            self.f = nn.functional.gelu
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu
        elif hidden_act.lower() == 'identity':
            self.f = lambda x: x
        elif hidden_act.lower() == 'one':
            def one(x):
                return torch.ones_like(x)
            self.f = one
        elif hidden_act.lower() == 'sigmoid':
            self.f = torch.sigmoid
        elif hidden_act.lower().startswith('rational:'):
            func_name = hidden_act.lower().split(':', 1)[1]
            self.f = Rational(
                cuda=f'cuda:{local_rank}', trainable=True, train_numerator=True,
                train_denominator=True, version="A", approx_func=func_name
            )

    def forward(self, x):
        # print("ADAPTER FORWARD")
        return self.f(x)


class PrintLayer(nn.Module):
    def __init__(self, id):
        super(PrintLayer, self).__init__()
        self.id = id

    def forward(self, x):
        # Do your print / debug stuff here
        if self.id:
            print("AFTER", x.isnan().any())
        else:
            print("BEFORE ACT", x.isnan().any())
        # print("BEFORE ACT", x.isnan().any())
        return x

    def backward(self, x):
        # print('backward', id, x)
        return x


class Adapter(nn.Module):

    def __init__(
        self,
        dim: int,
        downsample_factor: int = 4,
        hidden_act: str = "relu",
        add_layernorm: bool = False,
        adapter_switch: bool = False,
        initial_logits: list[float] = [0.5, 0.5],
        initial_temperature: float = 0.1,
        fixed_idx: int = None,

    ):
        super().__init__()
        layers = []
        if add_layernorm:
            layers.append(nn.LayerNorm(dim))
        layers.extend(
            [
                nn.Linear(dim, dim // downsample_factor),
                # nn.ReLU(),
                Activation_Function_Class(hidden_act),
                nn.Linear(dim // downsample_factor, dim),
            ]
        )
        local_rank, rank, world_size = get_world_info()
        self.local_rank = local_rank
        self.adapter_switch = adapter_switch
        device = f'cuda:{local_rank}'
        self.device = device
        if adapter_switch:
            self.switch_logits = nn.Parameter(torch.tensor(initial_logits))

            self.switch_temp = nn.Parameter(torch.tensor(initial_temperature))

            self.register_parameter("switch_logits", self.switch_logits)
            self.register_parameter("switch_temp", self.switch_temp)
            self.gumbel = torch.distributions.Gumbel(
                torch.tensor(0.), torch.tensor(1.))

            self.fixed_idx = fixed_idx
        self.adapter = nn.Sequential(*layers)

        self.adapter.apply(self.init_weights)

    def init_weights(self, m: nn.Module, std=1e-3):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=std)
            torch.nn.init.normal_(m.bias, std=std)
            m.weight.data = torch.clamp(
                m.weight.data, min=-2 * std, max=2 * std)
            m.bias.data = torch.clamp(m.bias.data, min=-2 * std, max=2 * std)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    def train(self, mode: bool = True):
        if self.adapter_switch:
            if not mode:
                self.fixed_idx = torch.argmax(
                    self.switch_logits, dim=-1).item()
            else:
                self.fixed_idx = None
        return super().train(mode)

    def forward(self, x: TensorType["b", "s", "d"]) -> TensorType["b", "s", "d"]:
        if self.adapter_switch:
            output = self.adapter(x)

            stacked = torch.stack((output, x), dim=2)

            batch_size, sequence_length, num_classes, hidden_dim_size = stacked.size()
            if not self.training:
                return stacked[:, :, self.fixed_idx, :]
            sample_size = [batch_size, num_classes]
            g = self.gumbel.sample(sample_size).to(
                device=self.device)

            weights = torch.softmax(
                (g + self.switch_logits)/self.switch_temp, dim=1).to(x.dtype)  # .half()

            y = torch.einsum('bsnd, bn -> bsd', stacked, weights)

            return y

        return self.adapter(x) + x


class ParallelAdapter(Adapter):
    def __init__(
        self,
        module: nn.Module,
        dim: int,
        downsample_factor: int = 4,
        scaled: bool = False,
        add_layernorm: bool = False,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__(
            dim, downsample_factor, add_layernorm=add_layernorm, activation=activation
        )
        self.module = module

        if scaled:
            # init scaling param
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.adapter_scale = 1

    def forward(self, x: TensorType["b", "s", "d"], **module_kwargs):
        y = self.module(x, **module_kwargs)
        z = self.adapter(x)
        return y + (z * self.adapter_scale)


class ParallelAdapterWrapper(ParallelAdapter):
    # used to add an adapter to the attention block

    def __init__(
        self,
        module: nn.Module,
        dim: int,
        downsample_factor: int = 4,
        scaled: bool = False,
        add_layernorm: bool = False,
        activation: nn.Module = nn.ReLU
    ):
        super().__init__(
            module, dim, downsample_factor, scaled, add_layernorm, activation
        )

    def forward(self, x: TensorType["b", "s", "d"], *attn_args, **attn_kwargs):
        attn_outputs = self.module(x, *attn_args, **attn_kwargs)
        attn_output, outputs = (
            attn_outputs[0],
            attn_outputs[1:],
        )  # output_attn: a, present, (attentions)
        hidden_states = attn_output + (self.adapter(x) * self.adapter_scale)
        return (hidden_states,) + outputs


class AdapterWrapper(Adapter):
    # used to add an adapter to the attention block

    def __init__(
        self,
        attn_block: nn.Module,
        dim: int,
        downsample_factor: int = 4,
        activation: nn.Module = nn.ReLU,
        add_layernorm: bool = False,

    ):
        super().__init__(dim, downsample_factor, activation, add_layernorm)
        self.attn_block = attn_block

    def forward(self, x: TensorType["b", "s", "d"], *attn_args, **attn_kwargs):
        attn_outputs = self.attn_block(x, *attn_args, **attn_kwargs)
        attn_output, outputs = (
            attn_outputs[0],
            attn_outputs[1:],
        )  # output_attn: a, present, (attentions)
        hidden_states = self.adapter(attn_output) + attn_output
        return (hidden_states,) + outputs
