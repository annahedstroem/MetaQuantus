from typing import Union, Callable, Optional, Any, Dict
import numpy as np
import gc
import torch

import json
import pickle
import scipy
import pathlib

from quantus.helpers import asserts
from quantus.helpers import utils
from quantus.helpers.model.model_interface import ModelInterface


def compute_iac_score(
    q: np.array,
    q_hat: np.array,
    indices: np.array,
    test_name: str,
    measure: Callable = scipy.stats.wilcoxon,
    alternative: str = "two-sided",
    zero_method: str = "zsplit",
    reverse_scoring: bool = True,
) -> float:
    """Compare evaluation scores by computing the p-value to test if the scores are statistically different.
    Returns p-value Wilcoxon Signed Rank test to see that the scores originates from different distributions.
    """
    assert isinstance(indices[0], np.bool_), "Indices must be of type bool."
    assert (
        q.ndim == 1 and q_hat.ndim == 1 and indices.ndim == 1
    ), "All inputs should be 1D."

    q = np.array(q)[np.array(indices)]
    q_hat = np.array(q_hat)[np.array(indices)]

    # if all(q == q_hat):
    #    return 1.0

    p_value = measure(q, q_hat, alternative=alternative, zero_method=zero_method)[1]

    if reverse_scoring and "Adversary" in analyser_name:
        return 1 - p_value

    return p_value


def compute_joint_p_value(
    pvals: np.array,
    method: str = "fisher",
    measure: Callable = scipy.stats.combine_pvalues,
) -> float:
    """Perform a Non-Parametric Combination (NPC) of existing pvalues
    from K perturbations - return a joint statistic in forms of a p-value."""
    return measure(np.array(pvals).flatten(), method=method)[1]


def compute_iec_adversary(
    Q_star: np.array,
    Q_hat: np.array,
    indices: np.array,
    lower_is_better: bool,
) -> float:
    """
    Return the mean of the agreement ranking matrix U \in [0, 1] to specify if the condition is met.

    Parameters
    ----------
    Q_star
    Q_hat
    indices
    lower_is_better

    Returns
    -------

    """
    U = []
    for row_star, row_hat in zip(Q_star, Q_hat):
        for q_star, q_hat in zip(row_star[indices], row_hat[indices]):

            if lower_is_better:
                if q_star < q_hat:
                    U.append(1)
                else:
                    U.append(0)
            else:
                if q_star > q_hat:
                    U.append(1)
                else:
                    U.append(0)
    return float(np.mean(U))


def compute_iec_resilience(
    Q_star: np.array, Q_hat: np.array, indices: np.array
) -> float:
    """
    Return the mean of the agreement ranking matrix U \in [0, 1] to specify if the condition is met.

    Parameters
    ----------
    Q_star
    Q_hat
    indices

    Returns
    -------

    """
    U = []

    # Sort the quality estimates of the different explanation methods.
    R_star = np.argsort(Q_star, axis=0)
    R_hat = np.argsort(Q_hat, axis=0)

    for ix, (row_star, row_hat) in enumerate(zip(R_star.T, R_hat.T)):
        if not indices[ix]:
            continue
        else:
            for q_star, q_hat in zip(row_star, row_hat):
                if q_star == q_hat:
                    U.append(1)
                else:
                    U.append(0)

    return float(np.mean(U))


def compute_iec_score(
    Q_star: np.array,
    Q_hat: np.array,
    indices: np.array,
    lower_is_better: bool,
    analyser_name: str,
) -> float:
    """

    Parameters
    ----------
    Q_star
    Q_hat
    indices
    lower_is_better
    analyser_name

    Returns
    -------

    """
    if "Adversary" in analyser_name:
        return compute_iec_adversary(
            Q_star=Q_star,
            Q_hat=Q_hat,
            indices=indices,
            lower_is_better=lower_is_better,
        )
    elif "Resilience" in analyser_name:
        return compute_iec_resilience(Q_star=Q_star, Q_hat=Q_hat, indices=indices)

    return np.nan


def generate_explanations(
    model: ModelInterface,
    x_batch: np.ndarray,
    y_batch: Optional[np.ndarray],
    explain_func: Callable,
    explain_func_kwargs: Optional[Dict],
    abs: Optional[bool],
    normalise: Optional[bool],
    normalise_func: Optional[Callable],
    normalise_func_kwargs: Optional[Dict],
    device: Optional[str],
) -> np.array:

    # Include device in explain_func_kwargs.
    if device is not None and "device" not in explain_func_kwargs:
        explain_func_kwargs["device"] = device

    # Asserts.
    asserts.assert_explain_func(explain_func=explain_func)

    # Generate explanations.
    a_batch = explain_func(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **explain_func_kwargs,
    )

    # Expand attributions to input dimensionality, asserts and inference of axes.
    a_batch = utils.expand_attribution_channel(a_batch, x_batch)
    asserts.assert_attributions(x_batch=x_batch, a_batch=a_batch)

    # Normalise with specified keyword arguments if requested.
    if normalise:
        a_batch = normalise_func(
            a=a_batch,
            normalise_axes=list(range(np.ndim(a_batch)))[1:],
            **normalise_func_kwargs,
        )

    # Take absolute if requested.
    if abs:
        a_batch = np.abs(a_batch)

    # Collect garbage.
    gc.collect()
    torch.cuda.empty_cache()

    return a_batch


def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return dict
    raise TypeError("The file is not a np.ndarray or dict. Not serializable")


def dump_obj(path: str, fname: str, obj: Any, use_json: bool = False) -> None:
    """Using pickle and json."""

    # Get path.
    full_name = str(path + fname).split("/")[:-1]
    full_path = ""
    for folder in full_name:
        full_path += folder
        full_path += "/"

    # Create folders if they don't exist.
    if not pathlib.Path(full_path).exists():
        print(f"Created a new folder for results {full_path[:-1]} to save {fname}.")
        try:
            pathlib.Path.mkdir(full_path[:-1], parents=True, exist_ok=True)
        except:
            print("It didn't work!")

    if use_json:
        with open(path + fname, "w") as f:
            json.dump(obj, f, default=default)
    else:
        with open(path + fname, "wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path: str, fname: Optional[str] = "", use_json: bool = False) -> Any:
    """Using pickle and json."""
    if use_json:
        with open(path + fname, "rb") as f:
            obj = json.load(f)  # , default=default)
    else:
        with open(path + fname, "rb") as f:
            obj = pickle.load(f)
    return obj


def get_statistics_intra_scores(
    intra_scores: dict, test: str, method: Optional[str] = None
):
    if method is None:
        intra_scores = np.array(list(intra_scores[test].values())).flatten()
        return (
            round(np.hstack(intra_scores).mean(), 4),
            round(np.hstack(intra_scores).std(), 4),
        )
    return (
        round(np.hstack(intra_scores[test][method]).mean(), 4),
        round(np.hstack(intra_scores[test][method]).std(), 4),
    )


def get_statistics_inter_scores(inter_scores: dict, test):
    return (
        round(np.hstack(inter_scores[test]).mean(), 4),
        round(np.hstack(inter_scores[test]).std(), 4),
    )
