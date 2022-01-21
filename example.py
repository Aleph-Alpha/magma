from multimodal_fewshot import Magma
from multimodal_fewshot.image import ImageInput

magma = Magma(
    checkpoint_path = 'mp_rank_00_model_states.pt', ## downloads automatically if not present in this path
    config_path = 'configs/MAGMA_v1.yml',
)
magma = magma.to('cuda:0') ## for some reason, this does not work on init

inputs =[
    ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
    'Describe the painting: A'
]

completion = magma.generate(inputs = inputs, num_tokens = 4, topk = 1) ## should return: "cabin on a lake"
print(f'completion: {completion}')