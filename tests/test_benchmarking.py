import pytest
import copy
import os
from typing import Union
from pytest_lazyfixture import lazy_fixture
import numpy as np

import torch
import quantus
from captum.attr import Saliency
import metaquantus
from metaquantus import setup_xai_settings, setup_estimators
from metaquantus import MetaEvaluation, MetaEvaluationBenchmarking

MINI_BATCH = 5

@pytest.mark.benchmarking
@pytest.mark.parametrize(
    "settings,explanation_methods,test_suite,estimator_category,expected",
    [(
            lazy_fixture("load_mnist_experimental_settings"),
            ["Gradient", "Saliency"],
            lazy_fixture("load_test_suite"),
            "Randomisation",
            {"min": 0, "max": 1},
        ),
    ],
)
def test_benchmarking_mnist(
    settings: dict,
    explanation_methods: list,
    test_suite: dict,
    estimator_category: str,
    expected: Union[float, dict, bool],
):
    dataset_name = "MNIST"

    # Load the experimental settings.
    dataset_settings, model_name, device = settings
    dataset_kwargs = dataset_settings["estimator_kwargs"]
    dataset_settings = {dataset_name: dataset_settings}

    # Get explanation methods.
    xai_methods = setup_xai_settings(
        xai_settings=explanation_methods,
        gc_layer=dataset_settings[dataset_name]["gc_layers"][model_name],
        img_size=dataset_kwargs["img_size"],
        nr_channels=dataset_kwargs["nr_channels"],
    )

    # Get estimators.
    estimators = setup_estimators(
        features=dataset_kwargs["features"],
        num_classes=dataset_kwargs["num_classes"],
        img_size=dataset_kwargs["img_size"],
        percentage=dataset_kwargs["percentage"],
        patch_size=dataset_kwargs["patch_size"],
        perturb_baseline=dataset_kwargs["perturb_baseline"],
    )
    estimators_category = {estimator_category: estimators[estimator_category]}

    ########################
    # Master run settings. #
    ########################

    # Set configs.
    iters = 5
    K = 3

    # Define the meta-evaluation exercise.
    meta_evaluator = MetaEvaluation(
        test_suite=test_suite,
        xai_methods=xai_methods,
        iterations=iters,
        nr_perturbations=K,
        write_to_file=False,
    )

    # Benchmark a category of metrics, using the intialised meta-evaluator.
    benchmark = MetaEvaluationBenchmarking(
        master=meta_evaluator,
        estimators=estimators_category,
        experimental_settings=dataset_settings,
        write_to_file=False,
        keep_results=True,
        channel_first=True,
        softmax=False,
        device=device,
        path=os.getcwd()+"tests/assets/results/",
        save=False,
    )()
    estimator_names = list(benchmark[dataset_name][model_name][estimator_category].keys())

    for estimator_name in estimator_names:

        scores_1 = np.array(
            list(benchmark[dataset_name][model_name][estimator_category][estimator_name][
            "results_meta_consistency_scores"
        ]["Model"]["consistency_results"].values()))
        scores_2 = np.array(
            list(benchmark[dataset_name][model_name][estimator_category][estimator_name][
            "results_meta_consistency_scores"
        ]["Input"]["consistency_results"].values()))

        assert np.all(
            ((scores_1 >= expected["min"]) & (scores_1 <= expected["max"])
             & (scores_2 >= expected["min"]) & (scores_2 <= expected["max"]))
        ), "Test failed."