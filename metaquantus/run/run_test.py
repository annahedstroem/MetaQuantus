import os
import warnings
import argparse
import torch
import numpy as np
from datetime import datetime

import quantus
from metaquantus.meta_evaluation import MetaEvaluation
from metaquantus.helpers.configs import (
    setup_estimators,
    setup_xai_methods,
    setup_dataset_models,
    setup_analyser_suite,
)
from metaquantus.helpers.utils import load_obj

PATH_ASSETS = "../assets/"
PATH_RESULTS = "/home/amlh/Projects/MetaQuantus/results/"

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
    args = parser.parse_args()

    dataset_name = str(args.dataset)
    K = int(args.K)
    iters = int(args.iters)
    print(dataset_name, K, iters)

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

    # Get explanation methods.
    xai_methods = setup_xai_methods(
        gc_layer=dataset_settings[dataset_name]["gc_layers"][model_name],
        img_size=dataset_kwargs["img_size"],
        nr_channels=dataset_kwargs["nr_channels"],
    )

    ########################
    # Master run settings. #
    ########################

    # Define metric.
    estimator_category = "Complexity"
    estimator_name = "Sparseness"

    # Define master!
    master = MetaEvaluation(
        analyser_suite=analyser_suite,
        xai_methods=xai_methods,
        iterations=iters,
        nr_perturbations=K,
        write_to_file=False,
    )

    master(
        estimator=estimators[estimator_category][estimator_name][0],
        model=dataset_settings[dataset_name]["models"][model_name],
        x_batch=dataset_settings[dataset_name]["x_batch"],
        y_batch=dataset_settings[dataset_name]["y_batch"],
        a_batch=None,
        s_batch=dataset_settings[dataset_name]["s_batch"],
        channel_first=True,
        softmax=False,
        device=device,
        lower_is_better=estimators[estimator_category][estimator_name][1],
    )
