"""This module contains different utilities used to support the meta-evaluation framework."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

import os
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
    """
    A supporting function to generate explanations.

    Parameters
    ----------
    model: torch.nn
        The model used in evaluation.
    x_batch: np.array
        The input data.
    y_batch: np.array
        The labels.
    explain_func: callable
        The function used for creating the explanation.
    explain_func_kwargs: dict
        The kwargs for each explanation method.
    abs: bool
        Indicates if an absolute operation is applied.
    normalise: bool
        Indicates if a normalisation operation is applied.
    normalise_func: callable
        The function used to normalise the explanations.
    normalise_func_kwargs: dict
        The kwargs for the normalisation function method.
    device: torch.device
        The device used, to enable GPUs.

    Returns
    -------
    np.array
    """

    # Collect garbage.
    gc.collect()
    torch.cuda.empty_cache()

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


def dump_obj(path: str, fname: str, obj: Any, use_json: bool = False) -> None:
    """
    Use pickle and json to dump an object.

    Parameters
    ----------
    path: str
        The path to dump the object.
    fname: str
        The filename.
    obj: Any
        The object to dump.
    use_json: bool
        Indicates if json where used when dumping the file.

    Returns
    -------
    obj
    """

    def default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return dict
        raise TypeError("The file is not a np.ndarray or dict. Not serializable")

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
    """
    Use pickle and json to load an object.

    Parameters
    ----------
    path: str
        The path to load the object.
    fname: str
        The filename.
    use_json: bool
        Indicates if json where used when dumping the file.

    Returns
    -------
    obj
    """
    if use_json:
        with open(path + fname, "rb") as f:
            obj = json.load(f)  # , default=default)
    else:
        with open(path + fname, "rb") as f:
            obj = pickle.load(f)
    return obj


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
    """
    Compare evaluation scores by computing the p-value to test if the scores are statistically different.
    Returns p-value Wilcoxon Signed Rank test to see that the scores originates from different distributions.

    Parameters
    ----------
    q: np.array
        An array of quality estimates.
    q_hat: np.array
        An array of perturbed quality estimates.
    indices: np.array
        The list of indices to perform the analysis on.
    test_name: string
        The type of test: either 'Adversary' or "'Resilience'.
    measure: callable
        A Callable such as scipy.stats.wilcoxon  or similar.
    alternative: string
        A string describing if it is two-sided or not.
    zero_method: string
        A string describing the method of how to treat zero differences.
    reverse_scoring: bool
        A boolean describing if reverse scoring should be applied.

    Returns
    -------
    float
    """
    assert isinstance(indices[0], np.bool_), "Indices must be of type bool."
    assert (
        q.ndim == 1 and q_hat.ndim == 1 and indices.ndim == 1
    ), "All inputs should be 1D."

    q = np.array(q)[np.array(indices)]
    q_hat = np.array(q_hat)[np.array(indices)]

    # Compute the p-value.
    p_value = measure(q, q_hat, alternative=alternative, zero_method=zero_method)[1]

    if reverse_scoring and "Adversary" in test_name:
        return 1 - p_value

    return p_value


def compute_iec_adversary(
    Q_star: np.array,
    Q_hat: np.array,
    indices: np.array,
    lower_is_better: bool,
) -> float:
    """
    Return the mean of the agreement ranking matrix U \in [0, 1] to specify if the ranking condition is met.

    Parameters
    ----------
    Q_star: np.array
        The matrix of quality estimates, averaged over nr_perturbations.
    Q_hat: np.array
        The matrix of perturbed quality estimates, averaged over nr_perturbations.
    indices: np.array
        The list of indices to perform the analysis on.
    lower_is_better: bool
        Indicates if lower values are considered better or not, to inverse the comparison symbol.

    Returns
    -------
    float
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
    Return the mean of the agreement ranking matrix U \in [0, 1] to specify if the ranking condition is met.

    Parameters
    ----------
    Q_star: np.array
        The matrix of quality estimates, averaged over nr_perturbations.
    Q_hat: np.array
        The matrix of perturbed quality estimates, averaged over nr_perturbations.
    indices: np.array
        The list of indices to perform the analysis on.

    Returns
    -------
    float
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
    test_name: str,
) -> float:
    """
    Return the mean of the agreement ranking matrix U \in [0, 1] to specify if the condition is met.

    Parameters
    ----------
    Q_star: np.array
        The matrix of quality estimates, averaged over nr_perturbations.
    Q_hat: np.array
        The matrix of perturbed quality estimates, averaged over nr_perturbations.
    indices: np.array
        The list of indices to perform the analysis on.
    lower_is_better: bool
        Indicates if lower values are considered better or not, to inverse the comparison symbol.
    test_name: string
        A string describing if the values is computed for 'Adversary' or 'Resilience'.
    Returns
    -------
    float
    """
    if "Adversary" in test_name:
        return compute_iec_adversary(
            Q_star=Q_star,
            Q_hat=Q_hat,
            indices=indices,
            lower_is_better=lower_is_better,
        )
    elif "Resilience" in test_name:
        return compute_iec_resilience(Q_star=Q_star, Q_hat=Q_hat, indices=indices)

    return np.nan
