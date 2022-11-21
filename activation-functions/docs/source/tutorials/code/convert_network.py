import torchvision

device = 'cuda:0'
vgg16 = torchvision.models.vgg16(pretrained=False).cuda(device)

from rational.utils.convert_network import convert_pytorch_model_to_rational
model = convert_pytorch_model_to_rational(vgg16, rational_cuda=True)
