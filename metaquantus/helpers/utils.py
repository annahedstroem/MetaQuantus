"""This module contains different utilities used to support the meta-evaluation framework."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

import os
from typing import Union, Callable, Optional, Any, Dict, Sequence
from importlib import util
import numpy as np
import gc
import torch
import json
import pickle
import scipy
import pathlib


def get_wrapped_model(
    model,
    channel_first: bool,
    softmax: bool,
    device: Optional[str] = None,
    model_predict_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Identifies the type of a model object and wraps the model in an appropriate interface.

    Source code: https://github.com/understandable-machine-intelligence-lab/Quantus

    Parameters
    ----------
    model: torch.nn.Module, tf.keras.Model
        A model this will be wrapped in the ModelInterface:
    channel_first: boolean, optional
         Indicates of the image dimensions are channel first, or channel last. Inferred from the input shape if None.
    softmax: boolean
        Indicates whether to use softmax probabilities or logits in model prediction. This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
    device: string
        Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
    model_predict_kwargs: dict, optional
        Keyword arguments to be passed to the model's predict method.

    Returns
    -------
    model
        A wrapped ModelInterface model.
    """
    if util.find_spec("torch"):
        if isinstance(model, torch.nn.Module):
            return PyTorchModel(
                model=model,
                channel_first=channel_first,
                softmax=softmax,
                device=device,
                model_predict_kwargs=model_predict_kwargs,
            )
    raise ValueError("Model needs to be torch.nn.Module.")


def expand_attribution_channel(a_batch: np.ndarray, x_batch: np.ndarray):
    """
    Expand additional channel dimension(s) for attributions if needed.

    Source code: https://github.com/understandable-machine-intelligence-lab/Quantus

    Parameters
    ----------
    x_batch: np.ndarray
        A np.ndarray which contains the input data that are explained.
    a_batch: np.ndarray
        An array which contains pre-computed attributions i.e., explanations.

    Returns
    -------
    np.ndarray
        A x_batch with dimensions matching those of a_batch.
    """
    if a_batch.shape[0] != x_batch.shape[0]:
        raise ValueError(
            f"a_batch and x_batch must have same number of batches ({a_batch.shape[0]} != {x_batch.shape[0]})"
        )
    if a_batch.ndim > x_batch.ndim:
        raise ValueError(
            f"a must not have greater ndim than x ({a_batch.ndim} > {x_batch.ndim})"
        )

    if a_batch.ndim == x_batch.ndim:
        return a_batch
    else:
        attr_axes = infer_attribution_axes(a_batch, x_batch)

        # TODO: Infer_attribution_axes currently returns dimensions w/o batch dimension.
        attr_axes = [a + 1 for a in attr_axes]
        expand_axes = [a for a in range(1, x_batch.ndim) if a not in attr_axes]

        return np.expand_dims(a_batch, axis=tuple(expand_axes))


def infer_attribution_axes(a_batch: np.ndarray, x_batch: np.ndarray) -> Sequence[int]:
    """
    Infers the axes in x_batch that are covered by a_batch.

    Source code: https://github.com/understandable-machine-intelligence-lab/Quantus

    Parameters
    ----------
    x_batch: np.ndarray
        A np.ndarray which contains the input data that are explained.
    a_batch: np.ndarray
        An array which contains pre-computed attributions i.e., explanations.

    Returns
    -------
    np.ndarray
        The axes inferred.
    """
    # TODO: Adapt for batched processing.

    if a_batch.shape[0] != x_batch.shape[0]:
        raise ValueError(
            f"a_batch and x_batch must have same number of batches ({a_batch.shape[0]} != {x_batch.shape[0]})"
        )

    if a_batch.ndim > x_batch.ndim:
        raise ValueError(
            "Attributions need to have <= dimensions than inputs, but {} > {}".format(
                a_batch.ndim, x_batch.ndim
            )
        )

    # TODO: We currently assume here that the batch axis is not carried into the perturbation functions.
    a_shape = [s for s in np.shape(a_batch)[1:] if s != 1]
    x_shape = [s for s in np.shape(x_batch)[1:]]

    if a_shape == x_shape:
        return np.arange(0, len(x_shape))

    # One attribution value per sample
    if len(a_shape) == 0:
        return np.array([])

    x_subshapes = [
        [x_shape[i] for i in range(start, start + len(a_shape))]
        for start in range(0, len(x_shape) - len(a_shape) + 1)
    ]
    if x_subshapes.count(a_shape) < 1:

        # Check that attribution dimensions are (consecutive) subdimensions of inputs
        raise ValueError(
            "Attribution dimensions are not (consecutive) subdimensions of inputs:  "
            "inputs were of shape {} and attributions of shape {}".format(
                x_batch.shape, a_batch.shape
            )
        )
    elif x_subshapes.count(a_shape) > 1:

        # Check that attribution dimensions are (unique) subdimensions of inputs.
        # Consider potentially expanded dims in attributions.

        if a_batch.ndim == x_batch.ndim and len(a_shape) < a_batch.ndim:
            a_subshapes = [
                [np.shape(a_batch)[1:][i] for i in range(start, start + len(a_shape))]
                for start in range(0, len(np.shape(a_batch)[1:]) - len(a_shape) + 1)
            ]
            if a_subshapes.count(a_shape) == 1:

                # Inferring channel shape.
                for dim in range(len(np.shape(a_batch)[1:]) + 1):
                    if a_shape == np.shape(a_batch)[1:][dim:]:
                        return np.arange(dim, len(np.shape(a_batch)[1:]))
                    if a_shape == np.shape(a_batch)[1:][:dim]:
                        return np.arange(0, dim)

            raise ValueError(
                "Attribution axes could not be inferred for inputs of "
                "shape {} and attributions of shape {}".format(
                    x_batch.shape, a_batch.shape
                )
            )

        raise ValueError(
            "Attribution dimensions are not unique subdimensions of inputs:  "
            "inputs were of shape {} and attributions of shape {}."
            "Please expand attribution dimensions for a unique solution".format(
                x_batch.shape, a_batch.shape
            )
        )
    else:
        # Infer attribution axes.
        for dim in range(len(x_shape) + 1):
            if a_shape == x_shape[dim:]:
                return np.arange(dim, len(x_shape))
            if a_shape == x_shape[:dim]:
                return np.arange(0, dim)

    raise ValueError(
        "Attribution axes could not be inferred for inputs of "
        "shape {} and attributions of shape {}".format(x_batch.shape, a_batch.shape)
    )


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
    a_batch = expand_attribution_channel(a_batch, x_batch)

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
