import torch
import torch.nn as nn
from typing import Literal, Optional, List, Callable, Union
from torchtyping import TensorType
import torch.nn.functional as F
from einops import rearrange


class PrefixAdapterv2(nn.Module):
    def __init__(
        self, dim, num_heads, attn_block, l=20, seq_length=2048, init_std=1e-3
    ):
        super().__init__()
        self.attn_block = attn_block
        assert dim % num_heads == 0
        self.adapter_Pk = nn.Parameter(
            torch.zeros(size=(1, num_heads, l, dim // num_heads))
        )  # (batch, head, seq_length, head_features)
        self.adapter_Pv = nn.Parameter(
            torch.zeros(size=(1, num_heads, l, dim // num_heads))
        )  # (batch, head, seq_length, head_features)
        self.l = l
        self.seq_length = seq_length
        self.init_weights(init_std)

    def init_weights(self, init_std=1e-3):
        # TODO: init with wte?
        torch.nn.init.normal_(self.adapter_Pk, std=init_std)
        torch.clamp(self.adapter_Pk.data, min=-2 * init_std, max=2 * init_std)
        torch.nn.init.normal_(self.adapter_Pv, std=init_std)
        torch.clamp(self.adapter_Pv.data, min=-2 * init_std, max=2 * init_std)

    def forward(self, x: TensorType["b", "s", "d"], *attn_args, **attn_kwargs):
        if attn_kwargs.get("layer_past") is not None:
            # we only want to append the prefix in the first inference step, so if there's already a layer past, just forward the model
            return self.attn_block(x, *attn_args, **attn_kwargs)
        else:
            # repeat along batch dim and pass through attn block
            attn_kwargs["layer_past"] = [
                i.repeat(x.shape[0], 1, 1, 1)
                for i in [self.adapter_Pk, self.adapter_Pv]
            ]
            return self.attn_block(x, *attn_args, **attn_kwargs)


class Adapter(nn.Module):
    def __init__(
        self,
        dim: int,
        downsample_factor: int = 4,
        activation: nn.Module = nn.ReLU,
        add_layernorm: bool = False,
    ):
        super().__init__()
        layers = []
        if add_layernorm:
            layers.append(nn.LayerNorm(dim))
        layers.extend(
            [
                nn.Linear(dim, dim // downsample_factor),
                activation(),
                nn.Linear(dim // downsample_factor, dim),
            ]
        )
        self.adapter = nn.Sequential(*layers)
        self.adapter.apply(self.init_weights)

    def init_weights(self, m: nn.Module, std=1e-3):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=std)
            torch.nn.init.normal_(m.bias, std=std)
            m.weight.data = torch.clamp(m.weight.data, min=-2 * std, max=2 * std)
            m.bias.data = torch.clamp(m.bias.data, min=-2 * std, max=2 * std)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    def forward(self, x: TensorType["b", "s", "d"]) -> TensorType["b", "s", "d"]:
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
        activation: nn.Module = nn.ReLU,
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


class PrefixAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        attention_module: nn.Module,
        q_proj: nn.Module,
        l: int = 20,
        num_heads: int = 8,
        split_heads_fn: Callable = None,
        merge_heads_fn: Callable = None,
        init_std=1e-3,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim ({dim}) should be divisible by num_heads ({num_heads})"
        head_dim = dim // num_heads

        # adapter params
        self.adapter_Pk = nn.Parameter(torch.zeros(l, head_dim))
        self.adapter_Pv = nn.Parameter(torch.zeros(l, head_dim))
        self.adapter_λ = nn.Parameter(torch.zeros(1))  # gating parameter

        self.softmax = nn.Softmax(dim=-1)
        self.attention_module = attention_module
        self.q_proj = q_proj
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.split_heads_fn = split_heads_fn or self._split_heads
        self.merge_heads_fn = merge_heads_fn or self._merge_heads
        self.init_weights(init_std)

    def init_weights(self, init_std=1e-3):
        # TODO: init with wte?
        torch.nn.init.normal_(self.adapter_Pk, std=init_std)
        torch.clamp(self.adapter_Pk.data, min=-2 * init_std, max=2 * init_std)
        torch.nn.init.normal_(self.adapter_Pv, std=init_std)
        torch.clamp(self.adapter_Pv.data, min=-2 * init_std, max=2 * init_std)
        torch.nn.init.uniform_(self.adapter_λ)

    def forward(self, x: TensorType["b", "s", "d"], *args, **kwargs):  # x = attn input
        attn_out = self.attention_module(x, *args, **kwargs)
        if isinstance(attn_out, tuple) and len(attn_out) >= 2:
            attn_out, other_outputs = attn_out[0], attn_out[1:]
        else:
            other_outputs = ()
        q = self.split_heads_fn(
            self.q_proj(x)
        )  # ideally we wouldn't have to run this projection twice, but i can't think of a better way to do it
        y = self.softmax(q @ self.adapter_Pk.T) @ self.adapter_Pv
        y = self.merge_heads_fn(y)
        z = (1 - self.adapter_λ) * attn_out + self.adapter_λ * y
        return (z, *other_outputs)

    def _split_heads(self, x: TensorType["b", "s", "d"]):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        if len(x.shape) == 5:
            return x.permute(
                0, 1, 3, 2, 4
            )  # (batch, blocks, head, block_length, head_features)
        elif len(x.shape) == 4:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(x.shape)}"
            )

    def _merge_heads(self, x: TensorType["b", "h", "s", "d"]):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        if len(x.shape) == 5:
            x = x.permute(0, 1, 3, 2, 4).contiguous()
        elif len(x.shape) == 4:
            x = x.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(
                f"Input tensor rank should be one of [4, 5], but is: {len(x.shape)}"
            )
        new_shape = x.size()[:-2] + (self.num_heads * self.head_dim,)
        return x.view(new_shape)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def init_weights(module, std=1e-3):
    """Initialize the weights."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # std defaults to 0.02, this might need to be changed
        module.weight.data.normal_(mean=0.0, std=std)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class MultiHeadedSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        downsample_dim=None,
        causal=True,
        dropout=0.1,
        pos_emb="none",
    ):
        super().__init__()

        self.pos_emb = pos_emb.lower()
        assert self.pos_emb in ["none"]  # TODO: add more options

        self.scale = dim ** -0.5
        self.heads = heads
        self.causal = causal

        if downsample_dim is not None:
            self.downsample_dim = downsample_dim
        else:
            self.downsample_dim = dim
        assert (
            self.downsample_dim % self.heads == 0
        ), f"{self.downsample_dim} % {self.heads} != 0"

        self.to_q = nn.Linear(dim, self.downsample_dim, bias=False)
        self.to_k = nn.Linear(dim, self.downsample_dim, bias=False)
        self.to_v = nn.Linear(dim, self.downsample_dim, bias=False)
        self.to_out = nn.Linear(self.downsample_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)

    def split_heads(self, x):
        return rearrange(x, "b s (h d) -> b h s d", h=self.heads)

    def merge_heads(self, x):
        return rearrange(x, "b h s d -> b s (h d)")

    def forward(self, x, mem=None):
        device = x.device
        q_in, k_in, v_in = x, x, x
        if mem is not None:
            k_in = torch.cat((mem, k_in), dim=-2)
            v_in = torch.cat((mem, v_in), dim=-2)
        q, k, v = self.to_q(q_in), self.to_k(k_in), self.to_v(v_in)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        dots = q @ k.transpose(-1, -2) / self.scale
        if self.causal:
            mask_value = max_neg_value(dots)
            i, j = dots.shape[-2:]
            r = torch.arange(i, device=device)
            mask = rearrange(r, "i -> () () i ()") < rearrange(r, "j -> () () () j")
            mask = F.pad(mask, (j - i, 0), value=False)
            dots.masked_fill_(mask, mask_value)
            del mask
        dots = F.softmax(dots, dim=-1)
        dots = self.dropout(dots)
        return self.to_out(self.merge_heads(dots @ v))


class AttentionAdapter(nn.Module):
    def __init__(
        self,
        dim,
        downsample_factor=12,
        heads=8,
        pos_emb="none",
        attn_dropout=0.1,
        ff_dropout=0.1,
    ):
        super().__init__()
        # x -> downsample proj -> attention -> mlp -> upsample -> residual
        downsample_dim = dim // downsample_factor

        self.adapter_ln1 = nn.LayerNorm(dim)
        self.adapter_attention = nn.Sequential(
            nn.LayerNorm(dim),
            MultiHeadedSelfAttention(
                dim=dim,
                downsample_dim=downsample_dim,
                heads=heads,
                pos_emb=pos_emb,
                dropout=attn_dropout,
            ),
        )
        self.adapter_ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, downsample_dim),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(downsample_dim, dim),
        )
        self.apply(init_weights)

    def forward(self, x):
        return x + self.adapter_attention(x) + self.adapter_ff(x)
