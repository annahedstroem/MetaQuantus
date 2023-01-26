import pytest
import pickle
import torch
import numpy as np

import metaquantus
from metaquantus import setup_dataset_models

@pytest.fixture(scope="session", autouse=True)
def load_cmnist_experimental_settings():
    """Load the experimental settings for cMNIST dataset."""

    dataset_name = "cMNIST"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_settings, model_name = setup_dataset_models(
        dataset_name=dataset_name, path_assets="tests/assets/", device=device
    )
    dataset_kwargs = dataset_settings[dataset_name]["estimator_kwargs"]

    return dataset_settings

@pytest.fixture(scope="session", autouse=True)
def load_cmnist_experimental_settings():
    """Load the experimental settings for cMNIST dataset."""

    dataset_name = "cMNIST"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_settings, model_name = setup_dataset_models(
        dataset_name=dataset_name, path_assets="tests/assets/", device=device
    )
    return dataset_settings["cMNIST"], model_name





"""python -m pytest"""