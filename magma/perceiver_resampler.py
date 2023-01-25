import torch
from torch import nn
from torchtyping import TensorType
from einops import rearrange
from einops_exts import rearrange_many


class PerceiverAttentionBlock(nn.Module):
    def __init__(
        self,
        token_dim,
        output_dim,
        num_heads=8,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temp = 1 / (output_dim ** -0.5)
        self.softmax = nn.Softmax(dim=-1)

        self.v_k_w = torch.nn.Linear(
            token_dim, output_dim * num_heads * 2, bias=False)
        self.q_w = torch.nn.Linear(
            token_dim, output_dim * num_heads, bias=False)
        self.out = nn.Linear(output_dim*num_heads, token_dim, bias=False)

    def forward(self, q: TensorType["Batch", "Number of Images", "OutputDim", "TokenDim"], k_v: TensorType["Batch", "Number of Images", "Sequence", "TokenDim"]):

        # (batch, n_images, sequence_length + output_dim, embedding_dim)
        k_v = torch.cat((k_v, q), dim=-2)
        # (batch, n_images, sequence_length + output_dim, embedding_dim)
        k, v = self.v_k_w(k_v).tensor_split(2, dim=-1)
        q = self.q_w(q)  # (batch, n_images, output_dim, embedding_dim)

        q, k, v = rearrange_many(
            (q, k, v), 'b n s (h d) -> b n h s d', h=self.num_heads)

        q = q * self.temp

        # k = Image Input Sequence Dimension / q = Output Dimension
        sm = torch.einsum("b n h q d,b n h k d->b n h q k", q, k)
        attn = sm.softmax(dim=-1)

        ff_input_per_head = torch.einsum(
            "b n h q k,b n h k d -> b n h q d", attn, v)
        ff_input = rearrange(
            ff_input_per_head, "b n h s d -> b n s (h d)", h=self.num_heads)
        return self.out(ff_input)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        token_dim,
        num_layers:  int = 2,
        n_latents: int = 64,
        time: int = 1,
    ):
        super().__init__()
        self.register_parameter("learned_latents", nn.Parameter(
            torch.randn(n_latents, token_dim)))
        self.register_parameter('time_embeddings', nn.Parameter(
            torch.rand(time, 1, token_dim)))
        self.flatten = torch.nn.Flatten()
        self.perceiver_attention_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.perceiver_attention_layers.append(nn.ModuleList([
                PerceiverAttentionBlock(
                    token_dim=token_dim,  output_dim=n_latents),
                PerceiverFeedForwardLayer(token_dim=token_dim)
            ]))
        self.normalize = nn.LayerNorm(token_dim)

    def forward(self, x: TensorType["Batch", "Number of Images", "Time", "Sequence", "Token Dimesion"]):
        if x.ndim == 3:
            x = x[:, None, None, :, :]

        batch_size, number_of_images, time, sequence_length, token_dim = x.size()
        x = rearrange(x, "b n t s d -> b n (t s) d")
        latents = self.learned_latents.repeat(
            batch_size, number_of_images, 1, 1)
        x = x + self.time_embeddings[:number_of_images]

        for att_module, ff_layer in self.perceiver_attention_layers:
            latents = att_module(latents, x)
            latents = ff_layer(latents)

        return self.normalize(latents)


class PerceiverFeedForwardLayer(nn.Module):
    def __init__(
        self,
        token_dim: int = 3027,
        mult: int = 4,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.inner = nn.Linear(token_dim, mult*token_dim, bias=False)
        self.act = nn.GELU()
        self.outer = nn.Linear(mult*token_dim, token_dim, bias=False)

    def forward(self, x):
        return self.outer(self.act(self.inner(self.norm(x))))
