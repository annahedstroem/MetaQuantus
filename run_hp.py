import os
import warnings
import argparse
import torch
import numpy as np
import pandas as pd
import uuid
from datetime import datetime

from metaquantus.master import MasterAnalyser
from metaquantus.utils import dump_obj
from metaquantus.configs import (
    setup_estimators,
    setup_xai_methods,
    setup_dataset_models,
    setup_analyser_suite,
)

PATH_ASSETS = "../assets/"
PATH_RESULTS = "/home/amlh/Projects/MetaQuantus/results/"

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
        args = parser.parse_args()

        dataset_name = str(args.dataset)
        K = int(args.K)
        iters = int(args.iters)

    except:
        dataset_name = "MNIST"
        K = 5
        iters = 2

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

    # Delete IPT, only run MPT.
    del analyser_suite["Input Resilience Test"]
    del analyser_suite["Input Adversary Test"]
    print(analyser_suite)

    # Get estimators.
    estimators = setup_estimators(
        features=dataset_kwargs["features"],
        num_classes=dataset_kwargs["num_classes"],
        img_size=dataset_kwargs["img_size"],
        percentage=dataset_kwargs["percentage"],
        perturb_baseline=dataset_kwargs["perturb_baseline"],
    )

    # Get explanation methods.
    xai_methods = setup_xai_methods(
        gc_layer=dataset_settings[dataset_name]["gc_layers"][model_name],
        img_size=dataset_kwargs["img_size"],
        nr_channels=dataset_kwargs["nr_channels"],
    )

    ##############################
    # Tuning exercise settings. #
    ##############################

    # Define master!
    master = MasterAnalyser(
        analyser_suite=analyser_suite,
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

                estimators[estimator_category][estimator_name][0].subset_size = s
                estimators[estimator_category][estimator_name][0].perturb_baseline = b
                estimators[estimator_category][estimator_name][0].nr_runs = n
                print(estimators[estimator_category][estimator_name][0].get_params)
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
        print("Could not hp experiment convert to dataframe.")
