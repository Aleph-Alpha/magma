import torch
import numpy as np
from multimodal_fewshot import Magma
from multimodal_fewshot.language_model import get_language_model
from multimodal_fewshot.utils import get_tokenizer

if __name__ == "__main__":
    # model = Magma.from_checkpoint(
    #     "configs/MAGMA_v1.yml",
    #     "/mnt/localdisk/mp_rank_00_model_states.pt",
    #     model_dir="/mnt/localdisk/gptj",
    #     lm_from_pretrained=True,
    # )
    # gptj_model = model.lm
    # model.half().cuda().eval()
    tokenizer = get_tokenizer()
    input_text = tokenizer.encode("this is a test", return_tensors="pt").cuda()
    input_img = torch.ones(1, 3, 384, 384).half().cuda()

    # input = model.embed([input_img, input_text])
    # logits = gptj_model(inputs_embeds=input).logits
    # logits = logits.detach().cpu().numpy()
    # np.save("/mnt/localdisk/logits_new.npy", logits)

    from transformers import GPTJForCausalLM
    import torch

    # load new model
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        revision="float16",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.cuda()

    model.eval()

    logits = model(input_text).logits
    logits = logits.detach().cpu().numpy()
    np.save("/mnt/localdisk/gptj_logits_new.npy", logits)

    print("test")
