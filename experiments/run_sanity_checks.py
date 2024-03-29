"""This module contains the script for obtaining the results associated with sanity checks experiment --- adversarial estimators."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

import os
import warnings
import argparse
import torch
import numpy as np
from datetime import datetime
import quantus

import quantus
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

    # Get estimators.
    estimators = setup_estimators(
        features=estimator_kwargs["features"],
        num_classes=estimator_kwargs["num_classes"],
        img_size=estimator_kwargs["img_size"],
        percentage=estimator_kwargs["percentage"],
        patch_size=estimator_kwargs["patch_size"],
        perturb_baseline=estimator_kwargs["perturb_baseline"],
    )
    estimators = {
        "Localisation": estimators["Localisation"],
        "Complexity": estimators["Complexity"],
        "Randomisation": estimators["Randomisation"],
        "Robustness": estimators["Robustness"],
        "Faithfulness": estimators["Faithfulness"],
    }

    # Get explanation methods.
    xai_methods = setup_xai_methods(
        gc_layer=dataset_settings[dataset_name]["gc_layers"][model_name],
        img_size=estimator_kwargs["img_size"],
        nr_channels=estimator_kwargs["nr_channels"],
    )

    ###########################
    # Sanity checks settings. #
    ###########################

    # Define master NR!
    master_nr = MetaEvaluation(
        test_suite=analyser_suite,
        xai_methods=xai_methods,
        iterations=iters,
        fname="Estimator_Different",
        nr_perturbations=K,
        sanity_check="Estimator_Different",
        path=PATH_RESULTS + "sanity_checks/",
    )

    master_nr(
        estimator=quantus.Metric,
        model=dataset_settings[dataset_name]["models"][model_name],
        x_batch=dataset_settings[dataset_name]["x_batch"],
        y_batch=dataset_settings[dataset_name]["y_batch"],
        a_batch=None,
        s_batch=dataset_settings[dataset_name]["s_batch"],
        channel_first=True,
        softmax=False,
        device=device,
    )

    uid_nr = master_nr.uid

    # Define master AR!
    master_ar = MetaEvaluation(
        test_suite=analyser_suite,
        xai_methods=xai_methods,
        iterations=iters,
        fname="Estimator_Same",
        nr_perturbations=K,
        sanity_check="Estimator_Same",
        path=PATH_RESULTS + "sanity_checks/",
    )

    master_ar(
        estimator=quantus.Metric,
        model=dataset_settings[dataset_name]["models"][model_name],
        x_batch=dataset_settings[dataset_name]["x_batch"],
        y_batch=dataset_settings[dataset_name]["y_batch"],
        a_batch=None,
        s_batch=dataset_settings[dataset_name]["s_batch"],
        channel_first=True,
        softmax=False,
        device=device,
    )

    uid_ar = master_ar.uid

    today = datetime.today().strftime("%d%m%Y")

    def print_mean_std(score_type: str, expectation: str, scores: np.array):
        if "IAC" in score_type:
            score_means = scores.mean(axis=(0, 2))
        else:
            score_means = scores.mean(axis=1)
        print(
            f"\t{score_type}={score_means.mean():.4f} ({score_means.std():.3f}) \t-----\tExpectation{expectation}"
        )

    for perturbation_type in ["Input", "Model"]:

        # uid_ar = "9b45"
        print(
            f"\nControlled scenario 1: the estimator always returns the same score, independent of perturbation (deterministic sampling). uid={uid_ar}\n"
        )
        print(f"{perturbation_type} Perturbation Test")

        inter_scores_nr = np.array(
            load_obj(
                PATH_RESULTS + "sanity_checks/",
                fname=f"{today}_Estimator_Same_inter_scores_{uid_ar}",
                use_json=True,
            )[f"{perturbation_type} Resilience Test"]
        ).reshape(iters, K)
        intra_scores_nr = load_obj(
            PATH_RESULTS + "sanity_checks/",
            fname=f"{today}_Estimator_Same_intra_scores_{uid_ar}",
            use_json=True,
        )
        intra_scores_nr = np.array(
            list(intra_scores_nr[f"{perturbation_type} Resilience Test"].values())
        ).reshape(len(xai_methods), iters, K)
        inter_scores_ar = np.array(
            load_obj(
                PATH_RESULTS + "sanity_checks/",
                fname=f"{today}_Estimator_Same_inter_scores_{uid_ar}",
                use_json=True,
            )[f"{perturbation_type} Adversary Test"]
        ).reshape(iters, K)
        intra_scores_ar = load_obj(
            PATH_RESULTS + "sanity_checks/",
            fname=f"{today}_Estimator_Same_intra_scores_{uid_ar}",
            use_json=True,
        )
        intra_scores_ar = np.array(
            list(intra_scores_ar[f"{perturbation_type} Adversary Test"].values())
        ).reshape(len(xai_methods), iters, K)

        print_mean_std(
            score_type="IAC_{NR}",
            expectation="=1.0 (should succeed: scores are the same!)",
            scores=intra_scores_nr,
        )
        print_mean_std(
            score_type="IAC_{AR}",
            expectation="=0.0 (should fail: scores are not different!)",
            scores=intra_scores_ar,
        )
        print_mean_std(
            score_type="IEC_{NR}",
            expectation="=1.0 (should succeed: scores, then rankings are the same!)",
            scores=inter_scores_nr,
        )
        print_mean_std(
            score_type="IEC_{AR}",
            expectation="=0.0 (should fail: does not fulfil ranking condition '<' since '=')",
            scores=inter_scores_ar,
        )

        # uid_nr = "908d"
        print(
            f"\nControlled scenario 2: the estimator always returns scores from a different distribution (stochastic sampling). uid={uid_nr}\n"
        )
        print(f"{perturbation_type} Perturbation Test")

        inter_scores_nr = np.array(
            load_obj(
                PATH_RESULTS + "sanity_checks/",
                fname=f"{today}_Estimator_Different_inter_scores_{uid_nr}",
                use_json=True,
            )[f"{perturbation_type} Resilience Test"]
        ).reshape(iters, K)
        intra_scores_nr = load_obj(
            PATH_RESULTS + "sanity_checks/",
            fname=f"{today}_Estimator_Different_intra_scores_{uid_nr}",
            use_json=True,
        )
        intra_scores_nr = np.array(
            list(intra_scores_nr[f"{perturbation_type} Resilience Test"].values())
        ).reshape(len(xai_methods), iters, K)
        inter_scores_ar = np.array(
            load_obj(
                PATH_RESULTS + "sanity_checks/",
                fname=f"{today}_Estimator_Different_inter_scores_{uid_nr}",
                use_json=True,
            )[f"{perturbation_type} Adversary Test"]
        ).reshape(iters, K)
        intra_scores_ar = load_obj(
            PATH_RESULTS + "sanity_checks/",
            fname=f"{today}_Estimator_Different_intra_scores_{uid_nr}",
            use_json=True,
        )
        intra_scores_ar = np.array(
            list(intra_scores_ar[f"{perturbation_type} Adversary Test"].values())
        ).reshape(len(xai_methods), iters, K)

        print_mean_std(
            score_type="IAC_{NR}",
            expectation="≈0.0 (should fail: scores are different!)",
            scores=intra_scores_nr,
        )
        print_mean_std(
            score_type="IAC_{AR}",
            expectation="≈1.0 (should succeed: scores are different!)",
            scores=intra_scores_ar,
        )
        print_mean_std(
            score_type="IEC_{NR}",
            expectation="≈0.25 (should be =1/L, where L=4: no diff in scores between explainers)",
            scores=inter_scores_nr,
        )
        print_mean_std(
            score_type="IEC_{AR}",
            expectation="≈0.0 (depends on the sampling distributions and its variation!)",
            scores=inter_scores_ar,
        )
