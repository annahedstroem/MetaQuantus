import pytest
import copy

from typing import Union
from pytest_lazyfixture import lazy_fixture
import numpy as np

import torch
import quantus
from captum.attr import Saliency
import metaquantus
from metaquantus import setup_xai_settings, setup_estimators
from metaquantus import MetaEvaluation

MINI_BATCH = 5


@pytest.mark.meta_evaluation
@pytest.mark.parametrize(
    "settings,explanation_methods,test_suite,estimator_category,estimator_name,expected",
    [
        (
            lazy_fixture("load_mnist_experimental_settings"),
            ["IntegratedGradients", "LayerGradCam"],
            lazy_fixture("load_test_suite"),
            "Localisation",
            "Pointing-Game",
            {"min": 0, "max": 1},
        ),
        (
            lazy_fixture("load_mnist_experimental_settings"),
            ["Saliency", "Gradient"],
            lazy_fixture("load_test_suite"),
            "Randomisation",
            "Random Logit",
            {"min": 0, "max": 1},
        ),
    ],
)
def test_meta_evaluation_mnist(
    settings: dict,
    explanation_methods: list,
    test_suite: dict,
    estimator_category: str,
    estimator_name: str,
    expected: Union[float, dict, bool],
):
    # Load the experimental settings.
    dataset_settings, model_name, device = settings
    dataset_kwargs = dataset_settings["estimator_kwargs"]
    model = dataset_settings["models"][model_name].eval()
    x_batch = dataset_settings["x_batch"][:MINI_BATCH]
    y_batch = dataset_settings["y_batch"][:MINI_BATCH]
    s_batch = dataset_settings["s_batch"][:MINI_BATCH]

    # Get explanation methods.
    xai_methods = setup_xai_settings(
        xai_settings=explanation_methods,
        gc_layer=dataset_settings["gc_layers"][model_name],
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

    ########################
    # Master run settings. #
    ########################

    # Set configs.
    iters = 5
    K = 3
    metric = estimators[estimator_category][estimator_name][0]
    lower_is_better = estimators[estimator_category][estimator_name][0]

    # Define the meta-evaluation exercise.
    meta_evaluator = MetaEvaluation(
        test_suite=test_suite,
        xai_methods=xai_methods,
        iterations=iters,
        nr_perturbations=K,
        write_to_file=False,
    )

    # Run the meta-evaluation.
    meta_evaluator(
        estimator=metric,
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=None,
        s_batch=s_batch,
        channel_first=True,
        softmax=False,
        device=device,
        lower_is_better=lower_is_better,
    )

    scores_1 = np.array(list(meta_evaluator.results_meta_consistency_scores["Model"]["consistency_results"].values()))
    scores_2 = np.array(list(meta_evaluator.results_meta_consistency_scores["Input"]["consistency_results"].values()))

    assert np.all(
        ((scores_1 >= expected["min"]) & (scores_1 <= expected["max"])
         & (scores_2 >= expected["min"]) & (scores_2 <= expected["max"]))
    ), "Test failed."


@pytest.mark.meta_evaluation
@pytest.mark.parametrize(
    "settings,explanation_methods,test_suite,estimator_category,estimator_name,expected",
    [
        (
            lazy_fixture("load_fmnist_experimental_settings"),
            ["IntegratedGradients", "LayerGradCam"],
            lazy_fixture("load_test_suite"),
            "Localisation",
            "Pointing-Game",
            {"min": 0, "max": 1},
        ),
        (
            lazy_fixture("load_mnist_experimental_settings"),
            ["Saliency", "Gradient"],
            lazy_fixture("load_test_suite"),
            "Randomisation",
            "Random Logit",
            {"min": 0, "max": 1},
        ),
    ],
)
def test_meta_evaluation_fmnist(
    settings: dict,
    explanation_methods: list,
    test_suite: dict,
    estimator_category: str,
    estimator_name: str,
    expected: Union[float, dict, bool],
):
    # Load the experimental settings.
    dataset_settings, model_name, device = settings
    dataset_kwargs = dataset_settings["estimator_kwargs"]
    model = dataset_settings["models"][model_name].eval()
    x_batch = dataset_settings["x_batch"][:MINI_BATCH]
    y_batch = dataset_settings["y_batch"][:MINI_BATCH]
    s_batch = dataset_settings["s_batch"][:MINI_BATCH]

    # Get explanation methods.
    xai_methods = setup_xai_settings(
        xai_settings=explanation_methods,
        gc_layer=dataset_settings["gc_layers"][model_name],
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

    ########################
    # Master run settings. #
    ########################

    # Set configs.
    iters = 5
    K = 10
    metric = estimators[estimator_category][estimator_name][0]
    lower_is_better = estimators[estimator_category][estimator_name][0]

    # Define the meta-evaluation exercise.
    meta_evaluator = MetaEvaluation(
        test_suite=test_suite,
        xai_methods=xai_methods,
        iterations=iters,
        nr_perturbations=K,
        write_to_file=False,
    )

    # Run the meta-evaluation.
    meta_evaluator(
        estimator=metric,
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=None,
        s_batch=s_batch,
        channel_first=True,
        softmax=False,
        device=device,
        lower_is_better=lower_is_better,
    )

    scores_1 = np.array(list(meta_evaluator.results_meta_consistency_scores["Model"]["consistency_results"].values()))
    scores_2 = np.array(list(meta_evaluator.results_meta_consistency_scores["Input"]["consistency_results"].values()))

    assert np.all(
        ((scores_1 >= expected["min"]) & (scores_1 <= expected["max"])
         & (scores_2 >= expected["min"]) & (scores_2 <= expected["max"]))
    ), "Test failed."


'''
@pytest.mark.skip(reason="torch gradient bug, unrelated metaquantus.")
@pytest.mark.meta_evaluation
@pytest.mark.parametrize(
    "settings,explanation_methods,test_suite,estimator_category,estimator_name,expected",
    [
        (
            lazy_fixture("load_cmnist_experimental_settings"),
            ["IntegratedGradients", "LayerGradCam"],
            lazy_fixture("load_test_suite"),
            "Localisation",
            "Pointing-Game",
            {"min": 0, "max": 1},
        ),
        (
            lazy_fixture("load_mnist_experimental_settings"),
            ["Saliency", "Gradient"],
            lazy_fixture("load_test_suite"),
            "Randomisation",
            "Random Logit",
            {"min": 0, "max": 1},
        ),

    ],
)
def test_meta_evaluation_cmnist(
    settings: dict,
    explanation_methods: list,
    test_suite: dict,
    estimator_category: str,
    estimator_name: str,
    expected: Union[float, dict, bool],
):
    # Load the experimental settings.
    dataset_settings, model_name, device = settings
    dataset_kwargs = dataset_settings["estimator_kwargs"]
    model = dataset_settings["models"][model_name].eval()
    x_batch = dataset_settings["x_batch"][:MINI_BATCH]
    y_batch = dataset_settings["y_batch"][:MINI_BATCH]
    s_batch = dataset_settings["s_batch"][:MINI_BATCH]

    # Get explanation methods.
    xai_methods = setup_xai_settings(
        xai_settings=explanation_methods,
        gc_layer=dataset_settings["gc_layers"][model_name],
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

    ########################
    # Master run settings. #
    ########################

    # Set configs.
    iters = 5
    K = 10
    metric = estimators[estimator_category][estimator_name][0]
    lower_is_better = estimators[estimator_category][estimator_name][0]

    # Define the meta-evaluation exercise.
    meta_evaluator = MetaEvaluation(
        test_suite=test_suite,
        xai_methods=xai_methods,
        iterations=iters,
        nr_perturbations=K,
        write_to_file=False,
    )

    # Run the meta-evaluation.
    meta_evaluator(
        estimator=metric,
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=None,
        s_batch=s_batch,
        channel_first=True,
        softmax=False,
        device=device,
        lower_is_better=lower_is_better,
    )

    scores_1 = np.array(list(meta_evaluator.results_meta_consistency_scores["Model"]["consistency_results"].values()))
    scores_2 = np.array(list(meta_evaluator.results_meta_consistency_scores["Input"]["consistency_results"].values()))

    assert np.all(
        ((scores_1 >= expected["min"]) & (scores_1 <= expected["max"])
         & (scores_2 >= expected["min"]) & (scores_2 <= expected["max"]))
    ), "Test failed."
'''