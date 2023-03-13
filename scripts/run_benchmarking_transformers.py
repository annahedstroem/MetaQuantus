"""This module contains the script for obtaining the results associated with the transformer benchmarking experiment."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

import os
import warnings
import argparse
import torch

from metaquantus import MetaEvaluation, MetaEvaluationBenchmarking
from metaquantus.helpers.configs import (
    setup_estimators_transformers,
    setup_xai_methods_transformers,
    setup_dataset_models_transformers,
    setup_test_suite,
)

PATH_ASSETS = "../assets/"
PATH_RESULTS = "results/"

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
    parser.add_argument("--start_idx")
    parser.add_argument("--end_idx")
    args = parser.parse_args()

    dataset_name = str(args.dataset)
    K = int(args.K)
    iters = int(args.iters)
    start_idx = int(args.start_idx)
    end_idx = int(args.end_idx)
    fname = f"{start_idx}-{end_idx}"
    print(dataset_name, K, iters, fname, start_idx, end_idx)

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
    SETTINGS, model_name = setup_dataset_models_transformers(
        dataset_name=dataset_name, path_assets=PATH_ASSETS, device=device
    )
    dataset_settings = {dataset_name: SETTINGS[dataset_name]}
    dataset_kwargs = dataset_settings[dataset_name]["estimator_kwargs"]

    # Get analyser suite.
    analyser_suite = setup_test_suite(dataset_name=dataset_name)

    # Get estimators.
    estimators = setup_estimators_transformers(
        features=dataset_kwargs["features"],
        num_classes=dataset_kwargs["num_classes"],
        img_size=dataset_kwargs["img_size"],
        percentage=dataset_kwargs["percentage"],
        patch_size=dataset_kwargs["patch_size"],
        perturb_baseline=dataset_kwargs["perturb_baseline"],
    )

    # Get explanation methods.
    xai_methods = setup_xai_methods_transformers(
        gc_layer=dataset_settings[dataset_name]["gc_layers"][model_name],
        img_size=dataset_kwargs["img_size"],
        nr_channels=dataset_kwargs["nr_channels"],
    )

    ###########################
    # Benchmarking settings. #
    ###########################

    # Reduce the number of samples.
    dataset_settings[dataset_name]["x_batch"] = dataset_settings[dataset_name]["x_batch"][start_idx:end_idx]
    dataset_settings[dataset_name]["y_batch"] = dataset_settings[dataset_name]["y_batch"][start_idx:end_idx]
    dataset_settings[dataset_name]["s_batch"] = dataset_settings[dataset_name]["s_batch"][start_idx:end_idx]

    # Define master!
    master = MetaEvaluation(
        test_suite=analyser_suite,
        xai_methods=xai_methods,
        iterations=iters,
        fname=fname,
        nr_perturbations=K,
    )

    # Benchmark!
    benchmark = MetaEvaluationBenchmarking(
        master=master,
        estimators=estimators,
        experimental_settings=dataset_settings,
        path=PATH_RESULTS,
        folder="benchmarks_new/",
        keep_results=True,
        channel_first=True,
        softmax=False,
        save=True,
        device=device,
    )()
