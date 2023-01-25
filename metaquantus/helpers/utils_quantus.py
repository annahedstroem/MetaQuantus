"""This module contains different utilities (private classes) sourced from Quantus library."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

from typing import Optional, Dict, Any, Sequence
import numpy as np


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
    if util.find_spec("tensorflow"):
        if isinstance(model, tf.keras.Model):
            return TensorFlowModel(
                model=model,
                channel_first=channel_first,
                softmax=softmax,
                model_predict_kwargs=model_predict_kwargs,
            )
    if util.find_spec("torch"):
        if isinstance(model, torch.nn.Module):
            return PyTorchModel(
                model=model,
                channel_first=channel_first,
                softmax=softmax,
                device=device,
                model_predict_kwargs=model_predict_kwargs,
            )
    raise ValueError("Model needs to be tf.keras.Model or torch.nn.Module.")




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