from typing import Dict
import os
import uuid
import warnings
import random
import argparse
from datetime import datetime
import gc

import torch
import numpy as np
import pandas as pd
import scipy

import metaquantus
from metaquantus import MetaEvaluation, MetaEvaluationBenchmarking
from metaquantus import dump_obj
from metaquantus import (
    setup_estimators,
    setup_complexity_estimators,
    setup_faithfulness_estimators,
    setup_randomisation_estimators,
    setup_xai_settings,
    setup_dataset_models,
    setup_analyser_suite,
)
import quantus

PATH_ASSETS = "../assets/"
PATH_RESULTS = "../results/"


if __name__ == "__main__":

    ######################
    # Parsing arguments. #
    ######################

    print(f"Running from path: {os.getcwd()}")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--fname")
    parser.add_argument("--K")
    parser.add_argument("--iters")
    parser.add_argument("--category")
    args = parser.parse_args()

    dataset_name = str(args.dataset)
    K = int(args.K)
    iters = int(args.iters)
    fname = str(args.fname)
    category = str(args.category)
    print(dataset_name, K, iters, fname, category)

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
    xai_setting_all = [
        "Gradient",
        "GradCAM",
        "GradientShap",
        "IntegratedGradients",
        "InputXGradient",
    ]

    def generate_random_explanation(model, inputs, targets, **kwargs):
        random_explanations = np.random.uniform(
            low=inputs.min(),
            high=inputs.max(),
            size=inputs.shape,
        )
        return random_explanations

    x_batch = dataset_settings[dataset_name]["x_batch"]
    y_batch = dataset_settings[dataset_name]["y_batch"]
    s_batch = dataset_settings[dataset_name]["s_batch"]

    df = pd.DataFrame()


    xai_setting = random.choices(xai_setting_all, k=3)
    xai_methods = setup_xai_settings(
        xai_settings=xai_setting,
        gc_layer=dataset_settings[dataset_name]["gc_layers"][model_name],
        img_size=dataset_kwargs["img_size"],
        nr_channels=dataset_kwargs["nr_channels"],
    )

    try:
        estimator_names = [e for e in estimators[category]]
        winner_is_same = (
            df.loc[
                (df["Estimator"] == estimator_names[0]) & (df["Rank"] == 1.0),
                "Method",
            ].values[0]
            == df.loc[
                (df["Estimator"] == estimator_names[1]) & (df["Rank"] == 1.0),
                "Method",
            ].values[0]
        )

    except:
        if not df.empty:
            print("...Winner method is the same.\n")

    uiid = uuid.uuid4()
    results = {}
    for method, kwargs in xai_methods.items():
        print(method)
        results[method] = {}

        model = dataset_settings[dataset_name]["models"]["ResNet9"].eval().cpu()

        random_explanations = generate_random_explanation(
            model=model, inputs=x_batch, targets=y_batch
        )

        scores_norm = {
            estimator_name: np.array(
                estimator_func[0](
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=None,
                    device=device,
                    explain_func=quantus.explain,
                    explain_func_kwargs={
                        **{
                            "method": method,
                        },
                        **kwargs,
                    },
                )
            )
            for estimator_name, estimator_func in estimators[category].items()
        }
        scores_ran = {
            estimator_name: np.array(
                estimator_func[0](
                    model=model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=random_explanations,
                    device=device,
                    explain_func=generate_random_explanation,
                )
            )
            for estimator_name, estimator_func in estimators[category].items()
        }

        for estimator_name in scores_norm.keys():
            results[method][estimator_name] = {
                "scores_norm": scores_norm[estimator_name],
                "scores_ran": scores_ran[estimator_name],
                "norm": np.linalg.norm(
                    scores_norm[estimator_name] - scores_ran[estimator_name]
                ),
                "p_val": scipy.stats.wilcoxon(
                    scores_norm[estimator_name], scores_ran[estimator_name]
                )[1],
                "p_corr": scipy.stats.pearsonr(
                    scores_norm[estimator_name], scores_ran[estimator_name]
                )[1],
                "s_corr": scipy.stats.spearmanr(
                    scores_norm[estimator_name], scores_ran[estimator_name]
                )[1],
            }

        # Collect garbage.
        gc.collect()
        torch.cuda.empty_cache()

        for (estimator_name, scores_n), (_, scores_r) in zip(
            scores_norm.items(), scores_ran.items()
        ):
            print(
                f"\t{estimator_name}: {np.nanmean(scores_n):.4f} ({np.std(scores_n):.2f}) {np.median(scores_n):.4f}"
            )

    # Save it.
    today = datetime.today().strftime("%d%m%Y")
    fname = f"example/{today}_{category.lower()}_ranking_exercise_{str(uiid)[:4]}"
    print(fname)
    dump_obj(
        obj=results,
        path=PATH_RESULTS,
        fname=fname,
        use_json=True,
    )

    try:

        df = pd.DataFrame(
            columns=["Estimator", "Method", "Faithfulness Score", "Rank"]
        )  # , "Score_std"])

        row = 0
        for mx, method in enumerate(results.keys()):
            for ex, estimator in enumerate(estimators[category]):
                row += mx + ex
                scores_n = results[method][estimator]["scores_norm"]
                df.loc[row, "Estimator"] = estimator
                df.loc[row, "Method"] = method
                df.loc[row, "Faithfulness Score"] = np.nanmean(scores_n)

        # Rank!
        df = df.sort_values(by="Estimator")
        df.index = np.arange(0, len(df))
        df.loc[: len(xai_methods), "Rank"] = df.groupby(["Estimator"])[
            "Faithfulness Score"
        ].rank(ascending=False)[: len(xai_methods)]
        df.loc[len(xai_methods) :, "Rank"] = df.groupby(["Estimator"])[
            "Faithfulness Score"
        ].rank(ascending=True)[len(xai_methods) :]
        df.to_csv(
            PATH_RESULTS
            + f"example/{today}_{category.lower()}_df_{str(uiid)[:4]}.csv"
        )
    except:
        print("Could not convert example experiment to dataframe.")