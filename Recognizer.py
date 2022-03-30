from myModel import myModel
import torch
from torch import nn
from torchvision import models

import numpy as np


def build_model():
    """
    Build the trained model
    :return: The trained model ready
    """
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 29)
    model.load_state_dict(torch.load("weights/vgg16.pt", map_location=torch.device("cpu")))
    return model


class recognizer(myModel):
    """
    A wrapper for the recognition model i trained to recognize letters in the american alphabet using
    sign language.
    """

    def __init__(self):
        self.model = build_model()
        self.model = self.model.float()
        self.model.eval()

    def process(self, image):
        image = np.swapaxes(image, 0, -1)
        image = image / 255.0  # normalize
        image = np.expand_dims(image, 0)
        # print(image.shape)
        inp = torch.from_numpy(image)
        return self.model(inp.float())
