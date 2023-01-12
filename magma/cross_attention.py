import torch
from torch import nn, tanh, einsum
from torchtyping import TensorType
from einops import rearrange, repeat
from einops_exts import rearrange_many
from .utils import get_world_info


# class CrossAttentionTransformerBlock(nn.Module):
#     def __init__(
#         self,
#         lm_block: nn.Module,
#         config: dict,
#         token_dim: int = 4096,
#         **kwargs
#     ):
#         super().__init__()
#         self.lm_block = lm_block
#         self.cross_x_block = GatedCrossAttentionBlock(
#             config, token_dim, **kwargs)
#         self.media_locations = None
#         self.visual_features = None

#     def forward(self, embs, **kwargs):
#         logits = self.cross_x_block(
#             embs, self.media_locations, self.visual_features)
#         out = self.lm_block(logits, use_cache=False, **kwargs)
#         return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: TensorType["Batch", "Sequence", "Dim"]):
        return self.net(x)


class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        text_token_dim: int = 4096,
        visual_token_dim: int = 2048,
        num_heads: int = 8,
        n_latents: int = 64, 
        head_dim: int = 64, 

    ):
        super().__init__()
        local_rank, rank, world_size = get_world_info()
        self.device = f'cuda:{local_rank}'
        self.num_heads = num_heads
        self.temp = 1 / (head_dim ** -0.5)
        self.softmax = nn.Softmax(dim=-1)


        self.v_k_w = nn.Linear(
            visual_token_dim, head_dim * num_heads * 2, bias=False)
        self.q_w = nn.Linear(
            text_token_dim, head_dim * num_heads, bias=False)
        self.out = nn.Linear(head_dim *num_heads, text_token_dim, bias=False)

    def forward(self,
                latent: TensorType["Batch", "Sequence", "TokenDim"],
                y: TensorType["Batch", "Sequence Length", "TokenDim"],
                media_mask: TensorType["Batch", "Sequence Length"]
                ):
        visual_features = rearrange(latent, 'b t n d -> b (t n) d')

        k, v = self.v_k_w(visual_features).chunk(2, dim=-1)
        q = self.q_w(y)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=8)
        sim = einsum('... i d, ... j d -> ... i j', q, k)
        media_time = torch.arange(3).to(self.device) + 1
        text_to_media_mask = rearrange(media_mask, 'b i -> b 1 i 1') == repeat(media_time, 'j -> 1 1 1 (j m)', m=64)
        sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)
        logits = sim.softmax(dim=-1)
        text_without_media_mask = media_mask == 0
        text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
        logits = logits.masked_fill(text_without_media_mask, 0.)
        logits = einsum('... i j, ... j d -> ... i d', logits, v)
        logits = rearrange(logits, 'b h n d -> b n (h d)')
        y = self.out(logits)
        return y


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, config, text_token_dim, visual_token_dim):
        super().__init__()
        self.x_attn = MaskedCrossAttention(
            text_token_dim = text_token_dim,
            visual_token_dim=visual_token_dim ,
            n_latents = config['n_latents']
            )
        self.tanh1 = nn.Parameter(torch.tensor([0.]))
        self.ffw = FeedForward(dim=text_token_dim)
        self.tanh2 = nn.Parameter(torch.tensor([0.]))

    def perceiver_pipe(self, visual_features, media_mask):
        self.media_mask = media_mask
        self.visual_features = visual_features

    def forward(self, embs: TensorType["Batch", "Sequence Length", "TokenDim"]):
        x_attn = self.x_attn(self.visual_features, embs, self.media_mask)
        attn_out = embs + tanh(self.tanh1) * x_attn
        x_ffw = attn_out + self.ffw(x_attn) * tanh(self.tanh2)

        return x_ffw
