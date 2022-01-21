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

## forward pass
embeddings = magma.preprocess_inputs(inputs = inputs)  ## returns a torch tensor of shape (1, sequence_length, hidden_dim)
outputs = magma(embeddings) ## output logits shape: torch.Size([1, 150, 50400])

## high level inference
completion = magma.generate(inputs = inputs, num_tokens = 4, topk = 1) ## completion: "cabin on a lake"