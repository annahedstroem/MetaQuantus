"""This module contains different functions to configure the experiments to perform meta-evaluation."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

from typing import Dict, List
import numpy as np
import torch

import quantus
from quantus import *
import torch
import torchvision

from .models import LeNet, ResNet9
from ..perturbation_tests.mpt import ModelPerturbationTest
from ..perturbation_tests.ipt import InputPerturbationTest


def setup_xai_methods(
    gc_layer: str,
    img_size: int = 28,
    nr_channels: int = 1,
) -> Dict:
    return {
        "Gradient": {},
        "Saliency": {},
        "IntegratedGradients": {},
        "LayerGradCam": {
            "gc_layer": gc_layer,
            "interpolate": (img_size, img_size),
            "interpolate_mode": "bilinear",
        },
        "Occlusion": {"window": (nr_channels, int(img_size / 4), int(img_size / 4))},
        "GradientShap": {},
    }


def setup_xai_settings(
    gc_layer: str,
    xai_settings: List[str],
    img_size: int = 28,
    nr_channels: int = 1,
) -> Dict:
    all_methods = {
        "Gradient": {},
        "Saliency": {},
        "IntegratedGradients": {},
        "LayerGradCam": {
            "gc_layer": gc_layer,
            "interpolate": (img_size, img_size),
            "interpolate_mode": "bilinear",
        },
        "Occlusion": {"window": (nr_channels, int(img_size / 4), int(img_size / 4))},
        "InputXGradient": {},
        "GradientShap": {},
        "Control Var. Random Uniform": {},
    }
    return {xai_method: all_methods[xai_method] for xai_method in xai_settings}


def setup_estimators(
    features: int,
    num_classes: int,
    img_size: int,
    percentage: int,
    patch_size: int,
    perturb_baseline: str = "uniform",
) -> Dict:
    return {
        "Robustness": {
            "Max-Sensitivity": (
                quantus.MaxSensitivity(
                    nr_samples=10,
                    perturb_func=perturb_func.uniform_noise,
                    norm_numerator=norm_func.fro_norm,
                    norm_denominator=norm_func.fro_norm,
                    lower_bound=0.01,
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                True,
            ),
            "Local Lipschitz Estimate": (
                quantus.LocalLipschitzEstimate(
                    nr_samples=10,
                    perturb_func=perturb_func.gaussian_noise,
                    norm_numerator=similarity_func.distance_euclidean,
                    norm_denominator=similarity_func.distance_euclidean,
                    perturb_std=0.01,
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                True,
            ),
        },
        "Randomisation": {
            "Random Logit": (
                quantus.RandomLogit(
                    similarity_func=similarity_func.correlation_spearman,
                    num_classes=num_classes,
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                True,
            ),
            "Model Parameter Randomisation Test": (
                quantus.ModelParameterRandomisation(
                    similarity_func=similarity_func.correlation_spearman,
                    return_sample_correlation=True,
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                True,
            ),
        },
        "Faithfulness": {
            "Faithfulness Correlation": (
                quantus.FaithfulnessCorrelation(
                    subset_size=features,
                    perturb_baseline=perturb_baseline,
                    perturb_func=perturb_func.baseline_replacement_by_indices,
                    nr_runs=10,
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                False,
            ),
            "Pixel-Flipping": (
                quantus.PixelFlipping(
                    features_in_step=features,
                    perturb_baseline=perturb_baseline,
                    perturb_func=perturb_func.baseline_replacement_by_indices,
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    return_auc_per_sample=True,
                    disable_warnings=True,
                ),
                False,
            ),
        },
        "Complexity": {
            "Sparseness": (
                quantus.Sparseness(
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                False,
            ),
            "Complexity": (
                quantus.Complexity(
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                True,
            ),
        },
        "Localisation": {
            "Pointing-Game": (
                quantus.PointingGame(
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                False,
            ),
            "Relevance Mass Accuracy": (
                quantus.RelevanceMassAccuracy(
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                False,
            ),
        },
    }


def setup_dataset_models(
    dataset_name: str,
    path_assets: str,
    device: torch.device,
):

    SETTINGS = {}

    if dataset_name == "MNIST":

        # Paths.
        path_mnist_model = path_assets + "models/mnist_lenet"
        path_mnist_assets = path_assets + "test_sets/mnist_test_set.npy"

        # Example for how to reload assets and models to notebook.
        model_mnist = LeNet()
        if device.type == "cpu":
            model_mnist.load_state_dict(
                torch.load(path_mnist_model, map_location=torch.device("cpu"))
            )
        else:
            model_mnist.load_state_dict(torch.load(path_mnist_model))

        assets_mnist = np.load(path_mnist_assets, allow_pickle=True).item()
        x_batch_mnist = assets_mnist["x_batch"]
        y_batch_mnist = assets_mnist["y_batch"]
        s_batch_mnist = assets_mnist["s_batch"]

        s_batch_mnist = s_batch_mnist.reshape(len(x_batch_mnist), 1, 28, 28)

        # Add to settings.
        SETTINGS["MNIST"] = {
            "x_batch": x_batch_mnist,
            "y_batch": y_batch_mnist,
            "s_batch": s_batch_mnist,
            "models": {"LeNet": model_mnist},
            "gc_layers": {"LeNet": "list(model.named_modules())[3][1]"},
            "estimator_kwargs": {
                "features": 28 * 2,
                "num_classes": 10,
                "img_size": 28,
                "percentage": 0.1,
                "nr_channels": 1,
                "patch_size": 28 * 2,
                "perturb_baseline": "uniform",
            },
        }
        model_name = "LeNet"

    elif dataset_name == "fMNIST":

        # Paths.
        path_fmnist_model = path_assets + "models/fmnist_lenet_model"
        path_fmnist_assets = path_assets + "test_sets/fmnist_test_set.npy"

        # Example for how to reload assets and models to notebook.
        model_fmnist = LeNet()
        if device.type == "cpu":
            model_fmnist.load_state_dict(
                torch.load(path_fmnist_model, map_location=torch.device("cpu"))
            )
        else:
            model_fmnist.load_state_dict(torch.load(path_fmnist_model))

        assets_fmnist = np.load(path_fmnist_assets, allow_pickle=True).item()
        x_batch_fmnist = assets_fmnist["x_batch"]
        y_batch_fmnist = assets_fmnist["y_batch"]
        s_batch_fmnist = assets_fmnist["s_batch"]

        # s_batch_fmnist = s_batch_fmnist.reshape(len(x_batch_fmnist), 1, 28, 28)

        # Add to settings.
        SETTINGS["fMNIST"] = {
            "x_batch": x_batch_fmnist,
            "y_batch": y_batch_fmnist,
            "s_batch": s_batch_fmnist,
            "models": {"LeNet": model_fmnist},
            "gc_layers": {"LeNet": "list(model.named_modules())[3][1]"},
            "estimator_kwargs": {
                "features": 28 * 2,
                "num_classes": 10,
                "img_size": 28,
                "percentage": 0.1,
                "nr_channels": 1,
                "patch_size": 28 * 2,
                "perturb_baseline": "uniform",
            },
        }

        model_name = "LeNet"

    elif dataset_name == "cMNIST":

        # Paths.
        path_cmnist_model = path_assets + "models/cmnist_resnet9.ckpt"
        path_cmnist_assets = path_assets + "test_sets/cmnist_test_set.npy"
        s_type = "box"

        # Example for how to reload assets and models to notebook.
        model_cmnist = ResNet9(nr_channels=3, nr_classes=10)
        if device.type == "cpu":
            model_cmnist.load_state_dict(
                torch.load(path_cmnist_model, map_location=torch.device("cpu"))
            )
        else:
            model_cmnist.load_state_dict(torch.load(path_cmnist_model))

        assets_cmnist = np.load(path_cmnist_assets, allow_pickle=True).item()
        x_batch_cmnist = assets_cmnist["x_batch"].detach().numpy()
        y_batch_cmnist = assets_cmnist["y_batch"].detach().numpy()
        s_batch_cmnist = assets_cmnist[f"s_batch_{s_type}"]

        s_batch_cmnist = s_batch_cmnist.reshape(len(x_batch_cmnist), 1, 32, 32)

        # Add to settings.
        SETTINGS["cMNIST"] = {
            "x_batch": x_batch_cmnist,
            "y_batch": y_batch_cmnist,
            "s_batch": s_batch_cmnist,
            "models": {"ResNet9": model_cmnist},
            "gc_layers": {"ResNet9": "list(model.named_modules())[1][1][-6]"},
            "estimator_kwargs": {
                "features": 32 * 2,
                "num_classes": 10,
                "img_size": 32,
                "percentage": 0.1,
                "nr_channels": 3,
                "patch_size": 32 * 2,
                "perturb_baseline": "uniform",
            },
        }
        model_name = "ResNet9"

    elif dataset_name == "ImageNet":
        pass

        # Paths.
        # path_imagenet_model = path_assets + "models/imagenet_resnet18_model"
        path_imagenet_assets = (
            path_assets + "test_sets/imagenet_test_set_6.npy"
        )  # imagenet_test_set.npy"

        # Example for how to reload assets and models to notebook.
        model_imagenet_resnet18 = torchvision.models.resnet18(pretrained=True)
        # model_imagenet_vgg16 = torchvision.models.vgg16(pretrained=True)
        # model_imagenet_alexnet = torchvision.models.alexnet(pretrained=True)

        assets_imagenet = np.load(path_imagenet_assets, allow_pickle=True).item()
        x_batch_imagenet = assets_imagenet["x_batch"][:150]
        y_batch_imagenet = assets_imagenet["y_batch"][:150]
        s_batch_imagenet = assets_imagenet["s_batch"][:150]

        s_batch_imagenet = s_batch_imagenet.reshape(len(x_batch_imagenet), 1, 224, 224)

        print(len(s_batch_imagenet))

        # Add to settings.
        SETTINGS["ImageNet"] = {
            "x_batch": x_batch_imagenet,
            "y_batch": y_batch_imagenet,
            "s_batch": s_batch_imagenet,
            "models": {
                "ResNet18": model_imagenet_resnet18,
                #        "VGG16": model_imagenet_vgg16,
            },
            "gc_layers": {
                "ResNet18": "list(model.named_modules())[61][1]",
                #        "VGG16": 'list(model_imagenet_vgg16.named_modules())[28][1]',
            },
            "estimator_kwargs": {
                "num_classes": 1000,
                "img_size": 224,
                "features": 224 * 4,
                "percentage": 0.1,
                "nr_channels": 3,
                "patch_size": 224 * 2,
            },
        }
        model_name = "ResNet18"

    else:
        raise ValueError(
            "Provide a supported dataset {'MNIST', 'fMNIST', 'cMNIST' and 'ImageNet'}."
        )

    return SETTINGS, model_name


def setup_analyser_suite(dataset_name: str):
    # Prepare analyser suite
    analyser_suite = {}

    if dataset_name in ["MNIST", "fMNIST", "cMNIST"]:

        analyser_suite = {
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

    elif dataset_name in ["ImageNet"]:

        analyser_suite = {
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

    else:
        raise ValueError(
            "Provide a supported dataset {'MNIST', 'fMNIST', 'cMNIST' and 'ImageNet'}."
        )

    return analyser_suite


def setup_faithfulness_estimators_full(
    features: int,
    patch_size: int,
    num_classes: int,
    img_size: int,
    percentage: int,
    perturb_baseline: str = "uniform",
) -> Dict:
    return {
        "Faithfulness": {
            "Faithfulness Correlation": (
                quantus.FaithfulnessCorrelation(
                    subset_size=features,
                    perturb_baseline=perturb_baseline,
                    perturb_func=perturb_func.baseline_replacement_by_indices,
                    nr_runs=10,
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    disable_warnings=True,
                ),
                False,
            ),
            "Pixel-Flipping": (
                quantus.PixelFlipping(
                    features_in_step=features,
                    perturb_baseline=perturb_baseline,
                    perturb_func=perturb_func.baseline_replacement_by_indices,
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    return_auc_per_sample=True,
                    disable_warnings=True,
                ),
                True,
            ),
            "MonotonicityCorrelation": (
                quantus.MonotonicityCorrelation(
                    nr_samples=10,
                    features_in_step=features,
                    perturb_baseline=perturb_baseline,
                    perturb_func=perturb_func.baseline_replacement_by_indices,
                    similarity_func=similarity_func.correlation_spearman,
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    disable_warnings=True,
                ),
                False,
            ),
            "Infidelity": (
                quantus.Infidelity(
                    perturb_baseline=perturb_baseline,
                    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
                    n_perturb_samples=10,
                    perturb_patch_sizes=[patch_size],
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    return_auc_per_sample=True,
                    disable_warnings=True,
                ),
                False,
            ),
        },
    }


def setup_complexity_estimators(
    features: int,
    patch_size: int,
    num_classes: int,
    img_size: int,
    percentage: int,
    perturb_baseline: str = "uniform",
) -> Dict:
    return {
        "Complexity": {
            "Sparseness": (
                quantus.Sparseness(
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                False,
            ),
            "Complexity": (
                quantus.Complexity(
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                True,
            ),
        }
    }


def setup_faithfulness_estimators(
    features: int,
    patch_size: int,
    num_classes: int,
    img_size: int,
    percentage: int,
    perturb_baseline: str = "uniform",
) -> Dict:
    return {
        "Faithfulness": {
            "Faithfulness Correlation": (
                quantus.FaithfulnessCorrelation(
                    subset_size=features,
                    perturb_baseline=perturb_baseline,
                    perturb_func=perturb_func.baseline_replacement_by_indices,
                    nr_runs=10,
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    disable_warnings=True,
                ),
                False,
            ),
            "Monotonicity Correlation": (
                quantus.MonotonicityCorrelation(
                    nr_samples=10,
                    features_in_step=features,
                    perturb_baseline=perturb_baseline,
                    perturb_func=perturb_func.baseline_replacement_by_indices,
                    similarity_func=similarity_func.correlation_spearman,
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    disable_warnings=True,
                ),
                False,
            ),
        }
    }


def setup_randomisation_estimators(
    features: int,
    patch_size: int,
    num_classes: int,
    img_size: int,
    percentage: int,
    perturb_baseline: str = "uniform",
) -> Dict:
    return {
        "Randomisation": {
            "Random Logit": (
                quantus.RandomLogit(
                    similarity_func=similarity_func.correlation_spearman,
                    num_classes=num_classes,
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                True,
            ),
            "Model Parameter Randomisation Test": (
                quantus.ModelParameterRandomisation(
                    similarity_func=similarity_func.correlation_spearman,
                    return_sample_correlation=True,
                    abs=False,
                    normalise=True,
                    normalise_func=normalise_func.normalise_by_average_second_moment_estimate,
                    return_aggregate=False,
                    aggregate_func=np.mean,
                    disable_warnings=True,
                ),
                True,
            ),
        },
    }
