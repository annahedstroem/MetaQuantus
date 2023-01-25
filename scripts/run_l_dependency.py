import os
import warnings
import argparse
import torch

import metaquantus
from metaquantus import MetaEvaluation
from metaquantus import MetaEvaluationBenchmarking
from metaquantus import (
    setup_estimators,
    setup_xai_settings,
    setup_dataset_models,
    setup_analyser_suite,
)

PATH_ASSETS = "../assets/"
PATH_RESULTS = "/home/amlh/Projects/MetaQuantus/results/"


def create_fname(xai_setting_name, estimators):
    """Create a name."""
    return (
        str(xai_setting_name)
        + "_"
        + str(list(estimators.keys()))
        .replace("[", "")
        .replace("]", "")
        .replace('"', "")
        .replace("'", "")
        .replace(", ", "_")
    )


if __name__ == "__main__":

    ######################
    # Parsing arguments. #
    ######################

    print(f"Running from path: {os.getcwd()}")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--K")
    parser.add_argument("--iters")
    parser.add_argument("--reversed_order")
    args = parser.parse_args()

    dataset_name = str(args.dataset)
    K = int(args.K)
    iters = int(args.iters)
    reversed_order = bool(args.reversed_order)
    print(dataset_name, K, iters, reversed_order)

    #########
    # GPUs. #
    #########

    # Setting device on GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()
    print(torch.version.cuda)

    # Additional info when using cuda.
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), "GB")

    ##############################
    # Dataset-specific settings. #
    ##############################

    # Get input, outputs settings.
    SETTINGS, model_name = setup_dataset_models(
        dataset_name=dataset_name, path_assets=PATH_ASSETS, device=device
    )
    dataset_settings = {dataset_name: SETTINGS[dataset_name]}
    dataset_kwargs = dataset_settings[dataset_name]["estimator_kwargs"]

    # Get analyser suite.
    analyser_suite = setup_analyser_suite(dataset_name=dataset_name)

    # Get estimators.
    estimators = setup_estimators(
        features=dataset_kwargs["features"],
        num_classes=dataset_kwargs["num_classes"],
        img_size=dataset_kwargs["img_size"],
        percentage=dataset_kwargs["percentage"],
        patch_size=dataset_kwargs["patch_size"],
        perturb_baseline=dataset_kwargs["perturb_baseline"],
    )
    estimators = {
        "Localisation": estimators["Localisation"],
        "Complexity": estimators["Complexity"],
        "Randomisation": estimators["Randomisation"],
        "Robustness": estimators["Robustness"],
        "Faithfulness": estimators["Faithfulness"],
    }

    ##########################
    # L dependency settings. #
    ##########################

    # Sets of 2, 3, 4.
    xai_settings = {
        "2_GR_SA": ["Gradient", "Saliency"],
        "2_GR_IG": ["Gradient", "IntegratedGradients"],
        "2_GR_OC": ["Gradient", "Occlusion"],
        "2_GR_IX": ["Gradient", "InputXGradient"],
        # "3_GR_SA_IG": ["Gradient", "Saliency", "IntegratedGradients"],
        # "3_GR_GC_GS": ["Gradient", "GradCAM", "GradientShap"],
        # "4_GR_SA_OC_LR": ["Gradient", "Saliency", "Occlusion", "GradCAM"],
        # "4_GR_SA_IX_GC": [
        #    "Gradient",
        #    "Saliency",
        #    "InputXGradient",
        #    "IntegratedGradients",
        # ],
    }

    if reversed_order:
        xai_settings = {
            "4_GR_SA_IX_GC": [
                "Gradient",
                "Saliency",
                "InputXGradient",
                "IntegratedGradients",
            ],
        }

    for xai_setting_name, xai_setting in xai_settings.items():

        print(xai_setting)

        # Get explanation methods.
        xai_methods = setup_xai_settings(
            xai_settings=xai_setting,
            gc_layer=dataset_settings[dataset_name]["gc_layers"][model_name],
            img_size=dataset_kwargs["img_size"],
            nr_channels=dataset_kwargs["nr_channels"],
        )

        # Define master!
        master = MetaEvaluation(
            analyser_suite=analyser_suite,
            xai_methods=xai_methods,
            iterations=iters,
            fname=create_fname(
                xai_setting_name=xai_setting_name, estimators=estimators
            ),
            nr_perturbations=K,
        )

        # Benchmark!
        benchmark = MetaEvaluationBenchmarking(
            master=master,
            estimators=estimators,
            experimental_settings=dataset_settings,
            path=PATH_RESULTS,
            folder="l_dependency/",
            keep_results=True,
            channel_first=True,
            softmax=False,
            device=device,
        )()
