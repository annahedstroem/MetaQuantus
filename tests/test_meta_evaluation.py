import pytest

from typing import Union
from pytest_lazyfixture import lazy_fixture
import numpy as np

import metaquantus
from metaquantus import setup_xai_settings

@pytest.mark.meta_evaluation
@pytest.mark.parametrize(
    "settings,explanation_methods,expected",
    [
        (
            lazy_fixture("load_cmnist_experimental_settings"),
            ["IntegratedGradients", "Gradient", "LayerGradCam"],
            {},
        ),
    ],
)
def test_meta_evaluation(
    settings: dict,
    explanation_methods: list,
    expected: Union[float, dict, bool],
):
    # Load the experimental settings.
    dataset_settings, model_name = settings
    model = dataset_settings["models"][model_name].eval()
    x_batch = dataset_settings["x_batch"]
    y_batch = dataset_settings["y_batch"]
    s_batch = dataset_settings["s_batch"]

    # Get explanation methods.
    xai_methods = setup_xai_settings(
        xai_settings=explanation_methods,
        gc_layer=dataset_settings["gc_layers"][model_name],
        img_size=dataset_settings["estimator_kwargs"]["img_size"],
        nr_channels=dataset_settings["estimator_kwargs"]["nr_channels"],
    )
    print(y_batch)

"""
init_params = params.get("init", {})
    call_params = params.get("call", {})

    if params.get("a_batch_generate", True):
        explain = call_params["explain_func"]
        explain_func_kwargs = call_params.get("explain_func_kwargs", {})
        a_batch = explain(
            model=model,
            inputs=x_batch,
            targets=y_batch,
            **explain_func_kwargs,
        )
    elif "a_batch" in data:
        a_batch = data["a_batch"]
    else:
        a_batch = None

    if "exception" in expected:
        with pytest.raises(expected["exception"]):
            scores = FaithfulnessCorrelation(**init_params)(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                **call_params,
            )[0]
        return

    scores = FaithfulnessCorrelation(**init_params)(
        model=model,
        x_batch=x_batch,
        y_batch=y_batch,
        a_batch=a_batch,
        **call_params,
    )[0]

    assert np.all(
        ((scores >= expected["min"]) & (scores <= expected["max"]))
    ), "Test failed."

"""