import copy

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models


class Model(nn.Module):
    def __init__(self, num_classes=4):
        super(Model, self).__init__()
        self.model = models.efficientnet_b0()

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)
        return x

    def save(self, path: str):
        torch.save(copy.deepcopy(self.state_dict()), path)

    def init(self, path: str = "model.pth"):
        self.load_state_dict(torch.load(path))

