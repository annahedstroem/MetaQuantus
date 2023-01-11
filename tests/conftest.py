import pytest
import pickle
import torch
import numpy as np
from keras.datasets import cifar10

from quantus.helpers.model.models import LeNet

CIFAR_IMAGE_SIZE = 32
MNIST_IMAGE_SIZE = 28
MINI_BATCH_SIZE = 8


@pytest.fixture(scope="session", autouse=True)
def load_mnist_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(
        torch.load("tests/assets/mnist", map_location="cpu", pickle_module=pickle)
    )
    return model

