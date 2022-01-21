from multimodal_fewshot import Magma
from multimodal_fewshot.image import ImageInput

magma = Magma(
    checkpoint_path = 'mp_rank_00_model_states.pt', 
    tokenizer_name = "gpt2", 
    config_path = 'configs/MAGMA_v1.yml',
)
magma = magma.to('cuda:0') ## for some reason, this does not work on init

inputs =[
    ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
    'Describe the painting: A'
]

completion = magma.generate(inputs = inputs, num_tokens = 4)
print(f'completion: {completion}')