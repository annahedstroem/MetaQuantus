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

from metaquantus.meta_evaluation import MetaEvaluation
from metaquantus.utils import dump_obj
from metaquantus.meta_evaluation_multiple import MetaEvaluationMultiple
from metaquantus.configs import (
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

    """
    ###########################
    # Benchmarking settings. #
    ###########################
    
    # Define master!
    master = MetaEvaluation(
        analyser_suite=analyser_suite,
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
        folder="example/",
        keep_results=True,
        channel_first=True,
        softmax=False,
        device=device,
    )()
    """

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

    # We have a clear winner (MC score lower. Make a selection.)
    # We hope that this one produces the lowest scores for a random explanation.
    # What is hard that they do not share the same bound and direct interpretation, unless both corr.
    winner_is_same = True
    df = pd.DataFrame()

    while winner_is_same:

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
            # normal_explanations = quantus.explain(model=model, inputs=x_batch, targets=y_batch, **{**{"method": method}, **kwargs})

            scores_norm = {
                estimator_name: np.array(
                    estimator_func[0](
                        model=model,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        a_batch=None,
                        # a_batch=normal_explanations,
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
                # print(f"{estimator_name}")
                # print(f"\tScores NORMAL: {np.nanmean(scores_n):.4f} ({np.std(scores_n):.2f})")
                # print(f"\tScores RAND: {np.nanmean(scores_r):.4f} ({np.std(scores_r):.2f})")
                # print(f'\tNorm: {results[method][estimator_name]["norm"]}')
                # print(f'\tP-val: {results[method][estimator_name]["p_val"]}')
                # print(f'\tP corr: {results[method][estimator_name]["p_corr"]}')
                # print(f'\tS corr: {results[method][estimator_name]["s_corr"]}')

            # plot_motivating_example(scores_norm, scores_ran, x_batch, random_explanations, gradient_explanations, method)

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
                    # print(f"\tScores {method} {estimator}: {np.nanmean(scores_n):.4f} ({np.std(scores_n):.2f})")
                    # if estimator == "Pixel-Flipping":
                    #    estimator_name = estimator + " (↓)"
                    # else:
                    #    estimator_name = estimator + " (↑)"
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

    print("...Winner method is different.\n")

"""
def plot_motivating_example(scores_norm: dict, 
                            scores_ran: dict,
                            x_batch: np.array,
                            random_explanations: np.array,
                            gradient_explanations: np.array,
                            method: str,
                            save: bool = False):
    fix, ax = plt.subplots(1, 3, figsize=(12, 4))

    text_1 = f'     FC (↑): {np.nanmean(scores_norm["Faithfulness Correlation"]):.4f} ({np.std(scores_norm["Faithfulness Correlation"]):.2f}) \n\
        PF (↓): {np.nanmean(scores_norm["Pixel-Flipping"]):.4f} ({np.std(scores_norm["Pixel-Flipping"]):.2f})'
    text_2 = f'     FC (↑): {np.nanmean(scores_ran["Faithfulness Correlation"]):.4f} ({np.std(scores_ran["Faithfulness Correlation"]):.2f}) \n\
        PF (↓): {np.nanmean(scores_ran["Pixel-Flipping"]):.4f} ({np.std(scores_ran["Pixel-Flipping"]):.2f})'

    ax[0].imshow(x_batch[0].reshape(28,28), cmap="gray")
    ax[1].imshow(gradient_explanations[0].reshape(28,28), cmap="seismic")
    ax[2].imshow(random_explanations[0].reshape(28,28), cmap="seismic")

    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    ax[0].set_title("Input", fontsize=20)
    ax[1].set_title(method, fontsize=20)
    ax[2].set_title("Random", fontsize=20)
    #ax[2].set_xlabel("Random", fontsize=20)
    ax[1].text(0, 34, text_1, fontsize=18)
    ax[2].text(0, 34, text_2, fontsize=18)

    plt.tight_layout()
    if save:
        plt.savefig(path_results+"plots/"+f"motivating_example_{dataset_name}.png", dpi=1000)
    plt.show()
    
def generate_random_explanation(model, inputs, targets, **kwargs):
    random_explanations = np.random.uniform(
    low=inputs.min(),
    high=inputs.max(),
    size=inputs.shape,
    )
    return random_explanations

"""
