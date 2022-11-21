from numpy import full
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader as DL
import torch.nn as nn
# import torch.functional as F
from activations.torch import ReLU
from torch import optim
import torch
from activations.torch.utils.activation_utils import  _get_toplevel_functions, GroupedActivations


class MnistCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.l1 = nn.Sequential(nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
            ),
            nn.MaxPool2d(kernel_size=2)
        )

        self.l2 = nn.Sequential(nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
            ),
            nn.MaxPool2d(2)
        )

        self.l3 = nn.Linear(32 * 7 * 7, 10)

        self.a1 = ReLU()
        self.a2 = ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)

        x = x.view(x.size(0), -1)
        output = self.l3(x)
        return output


def save_model(model, save_name):
    full_pth = f"./models/{save_name}"
    torch.save(model.state_dict(), full_pth)

def load_model(model_name, modeltype=MnistCNN, *args, **kwargs):
    model = modeltype(*args, **kwargs)
    model.load_state_dict(torch.load(f"./models/{model_name}"))
    return model


def train(epochs, model, trainDataLoader):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    lossf = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for (batch_image, batch_label) in trainDataLoader:
            output = model(batch_image)
            loss = lossf(output, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Loss at epoch {epoch} := {epoch_loss / len(trainDataLoader)}")


def eval(model, testDataLoader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testDataLoader:
            test_output = model(images)
            model_pred = torch.max(test_output, 1)[1]
            acc = (model_pred == labels).sum().item()
            correct += acc
            total += labels.size(0)
        print(f"Accuracy is {correct / total}")






train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True,
)
test_data = datasets.FashionMNIST(
    root = 'data',
    train = False,
    transform = ToTensor(),
    download=True
)

trainLaoder = DL(train_data, batch_size=64, shuffle=True)
testLoader = DL(test_data, batch_size=64, shuffle=True)


model = MnistCNN()

top_lvl_functions = _get_toplevel_functions(model)
networkGrouped = GroupedActivations(top_lvl_functions, "MnistCNN")

num_epochs = 1
# save and load if possible
networkGrouped.save_all_inputs()
networkGrouped.change_category("eval1")
train(1, model, trainLaoder)
networkGrouped.change_category("eval2")
eval(model, trainLaoder)
networkGrouped.show_all()
networkGrouped.save_all_inputs(saving=False)


""" ReLU.save_all_inputs(category_name="train")
train(num_epochs, model, trainLaoder)
ReLU.save_all_inputs(category_name="eval")
eval(model, testLoader)
ReLU.show_all()
ReLU.save_all_inputs(saving=False) """
