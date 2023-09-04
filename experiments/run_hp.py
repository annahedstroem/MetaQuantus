"""This module contains the script for obtaining the results associated with the hyperparameter optimisation application."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

import os
import warnings
import argparse
import torch
import numpy as np
import pandas as pd
import uuid
from datetime import datetime

from metaquantus import MetaEvaluation
from metaquantus.helpers.configs import *
from metaquantus.helpers.utils import load_obj


if __name__ == "__main__":

    ######################
    # Parsing arguments. #
    ######################

    print(f"Running from path: {os.getcwd()}")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset")
        parser.add_argument("--K")
        parser.add_argument("--iters")
        parser.add_argument("--PATH_ASSETS")
        parser.add_argument("--PATH_RESULTS")
        args = parser.parse_args()

        dataset_name = str(args.dataset)
        K = int(args.K)
        iters = int(args.iters)
        PATH_ASSETS = str(args.PATH_ASSETS)
        PATH_RESULTS = str(args.PATH_RESULTS)
        print(dataset_name, K, iters, PATH_ASSETS, PATH_RESULTS)

    except:
        dataset_name = "MNIST"
        K = 5
        iters = 2

    #########
    # GPUs. #
    #########

    # Setting device on GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)
    print("\t{torch.version.cuda}")

    # Additional info when using cuda.
    if device.type == "cuda":
        print(f"\t{torch.cuda.get_device_name(0)}")
        print("\tMemory Usage:")
        print("\tAllocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("\tCached:   ", round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), "GB")

    ##############################
    # Dataset-specific settings. #
    ##############################

    # Get input, outputs settings.
    SETTINGS, model_name = setup_dataset_models(
        dataset_name=dataset_name, path_assets=PATH_ASSETS, device=device
    )
    dataset_settings = {dataset_name: SETTINGS[dataset_name]}
    estimator_kwargs = dataset_settings[dataset_name]["estimator_kwargs"]

    # Get analyser suite.
    analyser_suite = setup_test_suite(dataset_name=dataset_name)

    # Delete IPT, only run MPT.
    del analyser_suite["Input Resilience Test"]
    del analyser_suite["Input Adversary Test"]
    print(analyser_suite)

    # Get estimators.
    estimators = setup_estimators(
        features=estimator_kwargs["features"],
        num_classes=estimator_kwargs["num_classes"],
        img_size=estimator_kwargs["img_size"],
        percentage=estimator_kwargs["percentage"],
        patch_size=estimator_kwargs["patch_size"],
        perturb_baseline=estimator_kwargs["perturb_baseline"],
    )

    # Get explanation methods.
    xai_methods = setup_xai_methods(
        gc_layer=dataset_settings[dataset_name]["gc_layers"][model_name],
        img_size=estimator_kwargs["img_size"],
        nr_channels=estimator_kwargs["nr_channels"],
    )

    ##############################
    # Tuning exercise settings. #
    ##############################

    # Define master!
    master = MetaEvaluation(
        test_suite=analyser_suite,
        xai_methods=xai_methods,
        iterations=iters,
        fname="hp",
        nr_perturbations=K,
        path=PATH_RESULTS,
    )

    result = {
        "Test": [],
        "MC Mean": [],
        "MC Std": [],
        "Nr Runs": [],
        "Baseline Strategy": [],
        "Subset Size": [],
    }

    # Define metric.
    estimator_category = "Faithfulness"
    estimator_name = "Faithfulness Correlation"

    # Define some parameter settings to evaluate.
    baseline_strategies = ["black", "uniform", "mean"]
    subset_sizes = np.array([52, 102, 128])
    nr_runs = [10, 25, 50]

    # Score explanations!
    i = 0
    for b in baseline_strategies:
        for s in subset_sizes:
            for n in nr_runs:

                estimators[estimator_category][estimator_name]["init"].subset_size = s
                estimators[estimator_category][estimator_name][
                    "init"
                ].perturb_baseline = b
                estimators[estimator_category][estimator_name]["init"].nr_runs = n
                print(estimators[estimator_category][estimator_name]["init"].get_params)
                master(
                    estimator=estimators[estimator_category][estimator_name]["init"],
                    model=dataset_settings[dataset_name]["models"][model_name],
                    x_batch=dataset_settings[dataset_name]["x_batch"],
                    y_batch=dataset_settings[dataset_name]["y_batch"],
                    a_batch=None,
                    s_batch=dataset_settings[dataset_name]["s_batch"],
                    channel_first=True,
                    softmax=False,
                    device=device,
                    score_direction=estimators[estimator_category][
                        estimator_name
                    ]["score_direction"],
                )

                for (
                    perturbation_type,
                    mc_results,
                ) in master.results_meta_consistency_scores.items():
                    if perturbation_type == "Input":
                        result["Test"].append("IPT")
                    else:
                        result["Test"].append("MPT")

                    result["MC Mean"].append(mc_results["MC_mean"])
                    result["MC Std"].append(mc_results["MC_std"])
                    if i == 0:
                        result = {
                            **result,
                            **{k: list() for k in mc_results["consistency_results"]},
                        }
                    for k, v in mc_results["consistency_results"].items():
                        result[k].append(v)
                    i += 1

                result["Baseline Strategy"].append(b.capitalize())
                result["Subset Size"].append(s)
                result["Nr Runs"].append(n)

                print(f"\niter={i+1} results={result}")

    for k, v in result.items():
        result[k] = np.array(v)

    # Save it.
    today = datetime.today().strftime("%d%m%Y")
    dump_obj(
        obj=result,
        path=PATH_RESULTS,
        fname=f"hp/{today}_hp_tuning_exercise_{str(uuid.uuid4())[:4]}",
        use_json=True,
    )

    try:
        df = pd.DataFrame(result)
        df.to_csv(
            PATH_RESULTS + f"hp/{today}_hp_tuning_exercise_{str(uuid.uuid4())[:4]}.csv"
        )
    except:
        print("Could not convert hp experiment to dataframe.")