import torch
import torch.nn as nn
from typing import Literal, Optional, List, Callable, Union
import torch.nn.functional as F
from multimodal_fewshot.utils import get_tokenizer, infer_checkpoint_path_from_config
from .language_model import get_language_model

# ------------------------- Classification wrapper class ----------------------------------


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes, classifier_type):
        super().__init__()

        if classifier_type == "linear":
            self.module = nn.Linear(hidden_size, num_classes)
        else:
            raise ValueError(f"invalid argument classifier_type = {classifier_type}")

    def forward(self, x):
        return self.module(x)


class MultimodalClassifier(MultimodalLM):
    def __init__(
        self,
        lm: nn.Module,
        tokenizer,
        config,
        device=None,
    ):
        super().__init__(lm, tokenizer, config, device)

        self.class_dict = self.config.class_dict
        self.num_classes = self.class_dict["num_classes"]
        self.classifier_type = self.class_dict.get("classifier_type", "linear")
        self.interface_type = self.class_dict.get("interface_type", "last_hidden_state")
        self.interface_position = self.class_dict.get("interface_position", -1)
        if self.class_dict.get("freeze_model", False):
            # TODO turn off dropout for the model if it is frozen?
            for p in self.model.parameters():
                p.requires_grad = False

        self.class_head = ClassificationHead(
            self.lm.config.hidden_size, self.num_classes, self.classifier_type
        )

    # x = captions, shape=[b, s]
    def build_weight_mask(self, x, num_imgs=1):
        """
        Builds a weight mask from text input x [b,s] => w [b, s, d] that gives the average or last non eos
        embedding after contraction with the hidden states.
        """

        w = (x != self.tokenizer.eos_token_id).long()

        # padding corresponding to the prefix length
        w_prefix = torch.zeros(
            x.shape[0],
            self.image_prefix_seq_len * num_imgs,
            device=self.device,
            dtype=torch.long,
        )

        if self.interface_type == "average_hidden_state":
            w = w / torch.sum(w, dim=1).unsqueeze(dim=1)
            w = w.to(device=self.device, dtype=torch.float16)

        elif self.interface_type == "last_hidden_state":
            w_shifted = torch.cat(
                [w[:, 1:], torch.zeros(x.shape[0], 1, device=self.device)], dim=1
            )
            w = (w - w_shifted).to(device=self.device, dtype=torch.float16)
            # if the mask technique works properly, we should only see one nonzero value in each batch of w. If not, something is wrong.
            assert all(w.count_nonzero(dim=1) == 1), "Something has gone wrong"

        # TODO: What to do about the case where in the last_hidden_state computation the index is larger than seq_len - img_seq_len so it gets pushed out in the next line?
        # append
        # prepend prefix padding
        w = torch.cat([w_prefix, w[:, : -self.image_prefix_seq_len * num_imgs]], dim=1)
        if all(w.sum(dim=1) == 0):
            print("Warning: Input length exceeded maximum sequence length")

        return w

    @classmethod
    def from_pretrained(
        cls,
        config,
        model_dir="/mnt/localdisk/models",
        tokenizer_name="gpt2",
        device=None,
    ):
        tokenizer = get_tokenizer(tokenizer_name)
        model = cls(
            get_language_model(
                config.lm_name, model_dir=model_dir, from_pretrained=False, no_init=True
            ),
            tokenizer,
            config,
            device,
        )

        ckpt_path = config.class_dict.get("pretrained_checkpoint")
        load_strict = False

        classification_ckpt_path = config.load
        if classification_ckpt_path is not None:
            # load latest checkpoint if one exists
            ckpt_path = infer_checkpoint_path_from_config(config)
            load_strict = True

        assert ckpt_path is not None
        print(f"loading multimodal transformer checkpoint...")
        state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))["module"]
        model.load_state_dict(state_dict, strict=load_strict)
        print(f"loaded multimodal transformer from checkpoint {ckpt_path}")

        return model

    def forward(self, images, captions, labels, return_probs=True):

        # images = [l_images, r_images] for nlvr2
        if not isinstance(images, list):
            images = [images]

        embeddings = self.embed(
            images + [captions[:, : -self.image_prefix_seq_len * len(images)]]
        )

        # embeddings = self.embed([images, captions[:, : -self.image_prefix_seq_len]])

        lm_out = self.lm(inputs_embeds=embeddings, output_hidden_states=True)

        hidden_states = lm_out.hidden_states[self.interface_position]

        w = self.build_weight_mask(captions, num_imgs=len(images))

        class_embeddings = torch.einsum("bsd, bs -> bd", hidden_states, w)

        logits = self.class_head(class_embeddings)

        loss = F.cross_entropy(logits, labels)

        if return_probs:
            return loss, F.softmax(logits, dim=1)

        return loss
