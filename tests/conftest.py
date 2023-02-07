import pytest
import pickle
import torch
import numpy as np

import metaquantus
from metaquantus import setup_dataset_models
from metaquantus import ModelPerturbationTest, InputPerturbationTest


@pytest.fixture(scope="session", autouse=True)
def load_cmnist_experimental_settings():
    """Load the experimental settings for cMNIST dataset."""

    dataset_name = "cMNIST"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_settings, model_name = setup_dataset_models(
        dataset_name=dataset_name, path_assets="tests/assets/", device=device
    )
    dataset_kwargs = dataset_settings[dataset_name]["estimator_kwargs"]

    return dataset_settings[dataset_name], model_name, device


@pytest.fixture(scope="session", autouse=True)
def load_mnist_experimental_settings():
    """Load the experimental settings for cMNIST dataset."""

    dataset_name = "MNIST"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_settings, model_name = setup_dataset_models(
        dataset_name=dataset_name, path_assets="tests/assets/", device=device
    )
    dataset_kwargs = dataset_settings[dataset_name]["estimator_kwargs"]

    return dataset_settings[dataset_name], model_name, device


@pytest.fixture(scope="session", autouse=True)
def load_fmnist_experimental_settings():
    """Load the experimental settings for cMNIST dataset."""

    dataset_name = "fMNIST"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_settings, model_name = setup_dataset_models(
        dataset_name=dataset_name, path_assets="tests/assets/", device=device
    )
    dataset_kwargs = dataset_settings[dataset_name]["estimator_kwargs"]

    return dataset_settings[dataset_name], model_name, device


@pytest.fixture(scope="session", autouse=True)
def load_test_suite():
    """Load the test suite for MNIST, fMNIST and cMNIST datasets."""

    test_suite = {
        "Model Resilience Test": ModelPerturbationTest(
            **{
                "noise_type": "multiplicative",
                "mean": 1.0,
                "std": 0.001,
                "type": "Resilience",
            }
        ),
        "Model Adversary Test": ModelPerturbationTest(
            **{
                "noise_type": "multiplicative",
                "mean": 1.0,
                "std": 2.0,
                "type": "Adversary",
            }
        ),
        "Input Resilience Test": InputPerturbationTest(
            **{
                "noise": 0.001,
                "type": "Resilience",
            }
        ),
        "Input Adversary Test": InputPerturbationTest(
            **{
                "noise": 5.0,
                "type": "Adversary",
            }
        ),
    }
    return test_suite