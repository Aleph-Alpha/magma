import torch
import torch.nn.functional as F
from torchtyping import TensorType
from typing import Union, List


def top_p_filter(logits: TensorType[..., "vocab"], threshold: float = 0.9):
    """
    Nucleus sampling
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs < (1 - threshold)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


def top_k_filter(logits, k):
    """
    Top K sampling
    """
    assert k > 0
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(1, ind, val)
    return probs


def remove_tokens_after_eos(tensor, eos_token, image_token):
    # any tokens after and end of sequence token is produced are also set to the eos token, and removed
    eos_index = (tensor == eos_token).nonzero()
    if eos_index.any():
        tensor[eos_index[0] :] = eos_token

    tensor = tensor.tolist()
    return [i for i in tensor if (not i == image_token) and (not i == eos_token)]


@torch.no_grad()
def generate(
    model: "Magma",
    embeddings: TensorType["b", "s", "d"],
    max_steps: int = 100,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 0.9,
    eos_token: int = None,
    decode: bool = True,
) -> Union[List[str], TensorType["b", "s"]]:
    """
    Generates captions for a batch of embeddings.

    :param model: The model to use for generation.
    :param embeddings: The embeddings to generate captions for.
    :param max_steps: The maximum number of steps to generate captions for.
    :param temperature: The temperature to use for sampling.
    :param top_k: value for top k sampling. If 0, no sampling will be used.
    :param top_p: value for top p sampling. If 0, no sampling will be used.
    :param eos_token: The token to use for end of sequence.
    :param decode: Whether to decode the output into text, or return the raw tokens.
    """

    # init values
    eos_token = eos_token or model.eos_token
    was_training = model.training
    model.eval()
    b, s, _ = embeddings.shape
    past_key_values = None

    # init output with image tokens
    out = torch.zeros((b, s), dtype=torch.long).to(model.device) + model.image_token

    # do sampling
    for i in range(max_steps):
        if i == 0:
            # initial input
            outputs = model.lm(
                inputs_embeds=embeddings,
                use_cache=True,
                past_key_values=past_key_values,
            )
        else:
            # now caching past k/v so we can use only the last token
            outputs = model.lm(
                input_ids=out[:, -1:], use_cache=True, past_key_values=past_key_values
            )

        logits = outputs.logits[:, -1, :].float()
        past_key_values = outputs.past_key_values

        # filter / temperature sample
        if temperature == 0.0:
            next_token = torch.argmax(logits, dim=-1)
        else:
            if top_k > 0:
                logits = top_k_filter(logits, k=top_k)
            if top_p > 0:
                logits = top_p_filter(logits, threshold=top_p)

            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        out = torch.cat((out, next_token), dim=-1)

        if eos_token is not None and (next_token == eos_token).all():
            break

    if decode:
        captions = []
        for b in out:
            b = remove_tokens_after_eos(b, eos_token, model.image_token)
            caption = model.tokenizer.decode(b)
            captions.append(caption)
        out = captions

    model.train(was_training)
    return out
