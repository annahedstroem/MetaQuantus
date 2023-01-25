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

from ..helpers.utils_quantus import expand_attribution_channel


def generate_explanations(
    model,
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

    # Generate explanations.
    a_batch = explain_func(
        model=model,
        inputs=x_batch,
        targets=y_batch,
        **explain_func_kwargs,
    )

    # Expand attributions to input dimensionality, asserts and inference of axes.
    a_batch = utils.expand_attribution_channel(a_batch, x_batch)

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
