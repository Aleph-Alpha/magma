from multimodal_fewshot.model import Magma
from multimodal_fewshot.image import ImageInput

magma = Magma(
    model_dir = '../multimodal_fewshot/model', 
    checkpoint_path = '../multimodal_fewshot/mp_rank_00_model_states.pt', 
    tokenizer_name = "gpt2", 
    config_path = '../multimodal_fewshot/configs/base.yml',
    lm_from_pretrained = False,
)
magma = magma.to('cuda:0') ## for some reason, this does not work on init

inputs =[
    ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
    'Describe the painting: A'
]

completion = magma.generate(inputs = inputs, num_tokens = 4)
print(f'completion: {completion}')