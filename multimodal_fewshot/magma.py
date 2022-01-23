import os
from os.path import exists
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MultimodalConfig
from .utils import get_tokenizer
from .transforms import get_transforms
from .sampling import top_k, top_p

from .model import (
    MultimodalLM, 
    get_language_model
)

## for downloading checkpoint
import gdown

class Magma(nn.Module):
    def __init__(
        self,  
        config_path,
        model_dir = './',
        tokenizer_name = 'gpt2', 
        lm_from_pretrained = False,
        device = 'cuda:0',
        checkpoint_path = 'mp_rank_00_model_states.pt',
    ):
        super().__init__()

        self.checkpoint_url = 'https://drive.google.com/u/0/uc?id=1EiAY3IcKWmGADaLDzdG25ykQghUwza6L&export=download'

        self.checkpoint_path = checkpoint_path
        checkpoint_exists = exists(self.checkpoint_path)

        if checkpoint_exists == False:
            print(f'model checkpoint does not exist in {self.checkpoint_path}\n')
            self.download_checkpoint(save_as = self.checkpoint_path)

        self.tokenizer = get_tokenizer(tokenizer_name)
        self.config = MultimodalConfig.from_yml(config_path)
        self.device = device

        self.model = MultimodalLM(
            lm=get_language_model(
                self.config.lm_name,
                model_dir=model_dir,
                from_pretrained=lm_from_pretrained,
                no_init=True,
            ),
            tokenizer= self.tokenizer,
            config= self.config,
        )

        self.image_transforms = get_transforms(self.config.image_size, model= self.model)

        if checkpoint_path is not None:
            sd = torch.load(checkpoint_path, map_location= self.device)
            self.model.load_state_dict(sd["module"])
        
        self.model.device = self.device

        self.model.half().eval()

    def download_checkpoint(self, save_as):
        '''
        Replace with something else later on when we host the model checkpoint somewhere else
        '''
        gdown.download(url = self.checkpoint_url, output = save_as, quiet=False)
              

    def preprocess_inputs(self, inputs): 
        """Converts a list of inputs into an embedding vector of shape (1, seq_len, hidden_dim)

        Args:
            inputs (list): list of inputs containing a combinations of strings and magma_explainer.utils.image.ImageInput 

        Returns:
            torch.tensor: an embedidng vector of shape (1, seq_len, hidden_dim)
        """

        list_of_tensors = []

        for i in range(len(inputs)):
            if isinstance(inputs[i], str):
                ## these are long tensors
                list_of_tensors.append(self.tokenizer.encode(inputs[i], return_tensors = 'pt'))
            else:
                transformed_image_tensor = self.image_transforms(inputs[i].get_image())
                assert transformed_image_tensor.ndim == 4 , f'Need a 4d tensor, but got {transformed_image.ndim} dimensions'
                list_of_tensors.append(transformed_image_tensor)

        return self.model.embed(list_of_tensors)

    def forward(
        self, 
        x, 
        output_hidden_states = False, 
        output_attentions = False, 
        past_key_values = None, 
        use_cache = False
    ):

        if isinstance(x, list):
            embeddings = self.preprocess_inputs(x.copy())
        else:
            embeddings = x
        
        output = self.model.lm(
            inputs_embeds= embeddings.to(self.device),
            use_cache = use_cache,
            past_key_values = past_key_values,
            output_hidden_states = output_hidden_states,
            output_attentions = output_attentions
        )

        return output

    @torch.no_grad()
    def generate(
        self, 
        inputs: list, 
        num_tokens: int, 
        temperature: float = 1., 
        filter_logits_fn = top_k,
        filter_threshold = 1.
    ):
        
        embeddings =  self.preprocess_inputs(inputs = inputs)

        output = self.model.generate(
            embeddings = embeddings,
            temperature=temperature,
            filter_threshold = filter_threshold,
            filter_logits_fn = filter_logits_fn,
            remove_tokens_after_eos= False,

        )[0]

        return output