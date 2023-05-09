"""This module contains the script for obtaining the results associated with the benchmarking experiment."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

import os
import warnings
import argparse
import torch
import torchvision
import timm

from metaquantus import MetaEvaluation, MetaEvaluationBenchmarking
from metaquantus.helpers.configs import *

if __name__ == "__main__":

    ######################
    # Parsing arguments. #
    ######################

    print(f"Running from path: {os.getcwd()}")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--fname", default="")
    parser.add_argument("--K", default=5)
    parser.add_argument("--iters", default=3)
    parser.add_argument("--batch_size", default=50)
    parser.add_argument("--reverse_order", default=False)
    parser.add_argument("--end_idx_fixed", default="")
    parser.add_argument("--start_idx_fixed", default="")
    parser.add_argument("--folder", default="benchmarks_imagenet/")
    parser.add_argument("--PATH_ASSETS", default="assets/")
    parser.add_argument("--PATH_RESULTS", default="results/")
    args = parser.parse_args()

    dataset_name = str(args.dataset)
    K = int(args.K)
    iters = int(args.iters)
    fname = str(args.fname)
    batch_size = int(args.batch_size)
    reverse_order = str(args.reverse_order)
    folder = str(args.folder)
    end_idx_fixed = eval(args.end_idx_fixed)
    start_idx_fixed = eval(args.start_idx_fixed)
    PATH_ASSETS = str(args.PATH_ASSETS)
    PATH_RESULTS = str(args.PATH_RESULTS)
    print(dataset_name, K, iters, batch_size, fname, reverse_order, folder, start_idx_fixed, end_idx_fixed, PATH_ASSETS, PATH_RESULTS)

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

    # Reduce the number of explanation methods and samples for ImageNet.
    if dataset_name == "ImageNet":
        setup_xai_methods = setup_xai_methods_imagenet
        setup_dataset_models = setup_dataset_models_imagenet_benchmarking

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
    analyser_suite = setup_test_suite(dataset_name=dataset_name)

    # Get estimators.
    estimators = setup_estimators(
        features=dataset_kwargs["features"],
        num_classes=dataset_kwargs["num_classes"],
        img_size=dataset_kwargs["img_size"],
        percentage=dataset_kwargs["percentage"],
        patch_size=dataset_kwargs["patch_size"],
        perturb_baseline=dataset_kwargs["perturb_baseline"],
    )

    estimators_sub = {
        "Complexity": estimators["Complexity"],
        "Localisation": estimators["Localisation"],
        "Randomisation": estimators["Randomisation"],
        "Robustness": estimators["Robustness"],
        "Faithfulness": estimators["Faithfulness"],
    }

    # Get explanation methods.
    xai_methods = setup_xai_methods(
        gc_layer=dataset_settings[dataset_name]["gc_layers"][model_name],
        img_size=dataset_kwargs["img_size"],
        nr_channels=dataset_kwargs["nr_channels"],
    )

    ###########################
    # Benchmarking settings. #
    ###########################

    if dataset_name != "ImageNet":

        # Define master!
        master = MetaEvaluation(
            test_suite=analyser_suite,
            xai_methods=xai_methods,
            iterations=iters,
            fname=fname,
            nr_perturbations=K,
            write_to_file=False,
        )

        # Benchmark!
        benchmark = MetaEvaluationBenchmarking(
            master=master,
            estimators=estimators_sub,
            experimental_settings=dataset_settings,
            path=PATH_RESULTS,
            folder=folder,
            keep_results=True,
            channel_first=True,
            softmax=False,
            save=True,
            device=device,
        )()

    else:

        # Retrieve the model.
        if fname == "ViT":
            dataset_settings[dataset_name]["models"] = {
                "ViT": torchvision.models.vit_b_16(pretrained=True),
            }
        elif fname == "ResNet18":
            dataset_settings[dataset_name]["models"] = {
                "ResNet18": torchvision.models.resnet18(pretrained=True),
            }
        elif fname == "Deit":
            dataset_settings[dataset_name]["models"] = {
                "Deit": timm.create_model(model_name='deit_tiny_distilled_patch16_224',
                                      pretrained=True),
            }

        # Prepare batching.
        nr_samples = len(dataset_settings[dataset_name]["x_batch"])
        indices_by_batch = list(range(0, nr_samples, batch_size))
        print(indices_by_batch, reverse_order)

        if eval(reverse_order):
            indices_by_batch = reversed(indices_by_batch)

        for start_idx in indices_by_batch:

            # Get indicies.
            end_idx = min(int(start_idx + batch_size), nr_samples)
            if (end_idx-start_idx) < batch_size:
                continue

            if end_idx_fixed:
                end_idx = end_idx_fixed
            if start_idx_fixed:
                start_idx = start_idx_fixed
            print(start_idx, end_idx)

            # Define master!
            master = MetaEvaluation(
                test_suite=analyser_suite,
                xai_methods=xai_methods,
                iterations=iters,
                fname=f"{fname}_{start_idx}:{end_idx}",
                nr_perturbations=K,
                write_to_file=False,
            )

            # Reduce the number of samples.
            dataset_settings[dataset_name]["x_batch"] = dataset_settings[dataset_name]["x_batch"][start_idx:end_idx]
            dataset_settings[dataset_name]["y_batch"] = dataset_settings[dataset_name]["y_batch"][start_idx:end_idx]
            dataset_settings[dataset_name]["s_batch"] = dataset_settings[dataset_name]["s_batch"][start_idx:end_idx]

            # Benchmark!
            benchmark = MetaEvaluationBenchmarking(
                master=master,
                estimators=estimators_sub,
                experimental_settings=dataset_settings,
                path=PATH_RESULTS,
                folder=folder,
                keep_results=True,
                channel_first=True,
                softmax=False,
                save=True,
                device=device,
            )()

            if start_idx_fixed is not None:
                break


