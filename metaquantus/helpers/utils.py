"""This module contains different utilities used to support the meta-evaluation framework."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

import os
from typing import Union, Callable, Optional, Any, Dict, Sequence, List, Tuple
from importlib import util
import numpy as np
import gc
import torch
import json
import pickle
import scipy
import pathlib
from contextlib import suppress
from copy import deepcopy
from abc import ABC, abstractmethod


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



def get_wrapped_model(
    model,
    channel_first: bool,
    softmax: bool,
    device: Optional[str] = None,
    model_predict_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Identifies the type of a model object and wraps the model in an appropriate interface.

    This function is not originally written by authors of MetaQuantus, but is sourced from the
    Quantus library as found at: https://github.com/understandable-machine-intelligence-lab/Quantus.

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

    This function is not originally written by authors of MetaQuantus, but is sourced from the
    Quantus library as found at: https://github.com/understandable-machine-intelligence-lab/Quantus.

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

    This function is not originally written by authors of MetaQuantus, but is sourced from the
    Quantus library as found at: https://github.com/understandable-machine-intelligence-lab/Quantus.

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



class ModelInterface(ABC):
    """Base ModelInterface for torch and tensorflow models."""

    def __init__(
        self,
        model,
        channel_first: bool = True,
        softmax: bool = False,
        model_predict_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialisation of ModelInterface class.

        This class is not originally written by authors of MetaQuantus, but is sourced from the
        Quantus library as found at: https://github.com/understandable-machine-intelligence-lab/Quantus.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A model this will be wrapped in the ModelInterface:
        channel_first: boolean, optional
             Indicates of the image dimensions are channel first, or channel last. Inferred from the input shape if None.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        """
        self.model = model
        self.channel_first = channel_first
        self.softmax = softmax

        if model_predict_kwargs is None:
            self.model_predict_kwargs = {}
        else:
            self.model_predict_kwargs = model_predict_kwargs

    @abstractmethod
    def predict(self, x: np.array, **kwargs):
        """
        Predict on the given input.

        Parameters
        ----------
        x: np.ndarray
         A given input that the wrapped model predicts on.
        kwargs: optional
            Keyword arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def shape_input(
        self,
        x: np.array,
        shape: Tuple[int, ...],
        channel_first: Optional[bool] = None,
        batched: bool = False,
    ):
        """
        Reshape input into model expected input.

        Parameters
        ----------
        x: np.ndarray
            A given input that is shaped.
        shape: Tuple[int...]
            The shape of the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        """
        Get the original torch/tf model.
        """
        raise NotImplementedError

    @abstractmethod
    def state_dict(self):
        """
        Get a dictionary of the model's learnable parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def get_random_layer_generator(self):
        """
        In every iteration yields a copy of the model with one additional layer's parameters randomized.
        For cascading randomization, set order (str) to 'top_down'. For independent randomization,
        set it to 'independent'. For bottom-up order, set it to 'bottom_up'.
        """
        raise NotImplementedError


    @abstractmethod
    def get_hidden_representations(
        self,
        x: np.ndarray,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Compute the model's internal representation of input x.
        In practice, this means, executing a forward pass and then, capturing the output of layers (of interest).
        As the exact definition of "internal model representation" is left out in the original paper (see: https://arxiv.org/pdf/2203.06877.pdf),
        we make the implementation flexible.
        It is up to the user whether all layers are used, or specific ones should be selected.
        The user can therefore select a layer by providing 'layer_names' (exclusive) or 'layer_indices'.

        Parameters
        ----------
        x: np.ndarray
            4D tensor, a batch of input datapoints
        layer_names: List[str]
            List with names of layers, from which output should be captured.
        layer_indices: List[int]
            List with indices of layers, from which output should be captured.
            Intended to use in case, when layer names are not unique, or unknown.

        Returns
        -------
        L: np.ndarray
            2D tensor with shape (batch_size, None)
        """
        raise NotImplementedError()


class PyTorchModel(ModelInterface):
    """Interface for torch models."""

    def __init__(
        self,
        model,
        channel_first: bool = True,
        softmax: bool = False,
        model_predict_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialisation of PyTorchModel class.

        This class is not originally written by authors of MetaQuantus, but is sourced from the
        Quantus library as found at: https://github.com/understandable-machine-intelligence-lab/Quantus.

        Parameters
        ----------
        model: torch.nn.Module, tf.keras.Model
            A model this will be wrapped in the ModelInterface:
        channel_first: boolean, optional
             Indicates of the image dimensions are channel first, or channel last. Inferred from the input shape if None.
        softmax: boolean
            Indicates whether to use softmax probabilities or logits in model prediction.
            This is used for this __call__ only and won't be saved as attribute. If None, self.softmax is used.
        model_predict_kwargs: dict, optional
            Keyword arguments to be passed to the model's predict method.
        device: string
            Indicated the device on which a torch.Tensor is or will be allocated: "cpu" or "gpu".
        """
        super().__init__(
            model=model,
            channel_first=channel_first,
            softmax=softmax,
            model_predict_kwargs=model_predict_kwargs,
        )
        self.device = device

    def predict(self, x: np.ndarray, grad: bool = False, **kwargs) -> np.array:
        """
        Predict on the given input.

        Parameters
        ----------
        x: np.ndarray
            A given input that the wrapped model predicts on.
        grad: boolean
            Indicates if gradient-calculation is disabled or not.
        kwargs: optional
            Keyword arguments.

        Returns
        --------
        np.ndarray
            predictions of the same dimension and shape as the input, values in the range [0, 1].
        """

        # Use kwargs of predict call if specified, but don't overwrite object attribute
        model_predict_kwargs = {**self.model_predict_kwargs, **kwargs}

        if self.model.training:
            raise AttributeError("Torch model needs to be in the evaluation mode.")

        grad_context = torch.no_grad() if not grad else suppress()

        with grad_context:
            pred = self.model(torch.Tensor(x).to(self.device), **model_predict_kwargs)
            if self.softmax:
                pred = torch.nn.Softmax(dim=-1)(pred)
            if pred.requires_grad:
                return pred.detach().cpu().numpy()
            return pred.cpu().numpy()

    def shape_input(
        self,
        x: np.array,
        shape: Tuple[int, ...],
        channel_first: Optional[bool] = None,
        batched: bool = False,
    ) -> np.array:
        """
        Reshape input into model expected input.

        Parameters
        ----------
        x: np.ndarray
             A given input that is shaped.
        shape: Tuple[int...]
            The shape of the input.
        channel_first: boolean, optional
            Indicates of the image dimensions are channel first, or channel last.
            Inferred from the input shape if None.
        batched: boolean
            Indicates if the first dimension should be expanded or not, if it is just a single instance.

        Returns
        -------
        np.ndarray
            A reshaped input.
        """
        if channel_first is None:
            channel_first = utils.infer_channel_first(x)

        # Expand first dimension if this is just a single instance.
        if not batched:
            x = x.reshape(1, *shape)

        # Set channel order according to expected input of model.
        if self.channel_first:
            return utils.make_channel_first(x, channel_first)
        raise ValueError("Channel first order expected for a torch model.")

    def get_model(self) -> torch.nn:
        """
        Get the original torch model.
        """
        return self.model

    def state_dict(self) -> dict:
        """
        Get a dictionary of the model's learnable parameters.
        """
        return self.model.state_dict()

    def get_random_layer_generator(self, order: str = "top_down", seed: int = 42):
        """
        In every iteration yields a copy of the model with one additional layer's parameters randomized.
        For cascading randomization, set order (str) to 'top_down'. For independent randomization,
        set it to 'independent'. For bottom-up order, set it to 'bottom_up'.

        Parameters
        ----------
        order: string
            The various ways that a model's weights of a layer can be randomised.
        seed: integer
            The seed of the random layer generator.

        Returns
        -------
        layer.name, random_layer_model: string, torch.nn
            The layer name and the model.
        """
        original_parameters = self.state_dict()
        random_layer_model = deepcopy(self.model)

        modules = [
            l
            for l in random_layer_model.named_modules()
            if (hasattr(l[1], "reset_parameters"))
        ]

        if order == "top_down":
            modules = modules[::-1]

        for module in modules:
            if order == "independent":
                random_layer_model.load_state_dict(original_parameters)
            torch.manual_seed(seed=seed + 1)
            module[1].reset_parameters()
            yield module[0], random_layer_model

    def sample(
        self,
        mean: float,
        std: float,
        noise_type: str = "multiplicative",
    ) -> torch.nn:
        """
        Sample a model by means of adding normally distributed noise.

        Parameters
        ----------
        mean: float
            The mean point to sample from.
        std: float
            The standard deviation to sample from.
        noise_type: string
            Noise type could be either 'additive' or 'multiplicative'.

        Returns
        -------
        model_copy: torch.nn
            A noisy copy of the orginal model.
        """

        distribution = torch.distributions.normal.Normal(loc=mean, scale=std)
        original_parameters = self.state_dict()
        model_copy = deepcopy(self.model)
        model_copy.load_state_dict(original_parameters)

        # If std is not zero, loop over each layer and add Gaussian noise.
        if not std == 0.0:
            with torch.no_grad():
                for layer in model_copy.parameters():
                    if noise_type == "additive":
                        layer.add_(distribution.sample(layer.size()).to(layer.device))
                    elif noise_type == "multiplicative":
                        layer.mul_(distribution.sample(layer.size()).to(layer.device))
                    else:
                        raise ValueError(
                            "Set noise_type to either 'multiplicative' "
                            "or 'additive' (string) when you sample the model."
                        )
        return model_copy


    def get_hidden_representations(
        self,
        x: np.ndarray,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> np.ndarray:

        """
        Compute the model's internal representation of input x.
        In practice, this means, executing a forward pass and then, capturing the output of layers (of interest).
        As the exact definition of "internal model representation" is left out in the original paper (see: https://arxiv.org/pdf/2203.06877.pdf),
        we make the implementation flexible.
        It is up to the user whether all layers are used, or specific ones should be selected.
        The user can therefore select a layer by providing 'layer_names' (exclusive) or 'layer_indices'.

        Parameters
        ----------
        x: np.ndarray
            4D tensor, a batch of input datapoints
        layer_names: List[str]
            List with names of layers, from which output should be captured.
        layer_indices: List[int]
            List with indices of layers, from which output should be captured.
            Intended to use in case, when layer names are not unique, or unknown.

        Returns
        -------
        L: np.ndarray
            2D tensor with shape (batch_size, None)
        """

        device = self.device if self.device is not None else "cpu"
        all_layers = [*self.model.named_modules()]
        num_layers = len(all_layers)

        if layer_indices is None:
            layer_indices = []

        # E.g., user can provide index -1, in order to get only representations of the last layer.
        # E.g., for 7 layers in total, this would correspond to positive index 6.
        positive_layer_indices = [
            i if i >= 0 else num_layers + i for i in layer_indices
        ]

        if layer_names is None:
            layer_names = []

        def is_layer_of_interest(layer_index: int, layer_name: str):
            if layer_names == [] and positive_layer_indices == []:
                return True
            return layer_index in positive_layer_indices or layer_name in layer_names

        # skip modules defined by subclassing API.
        hidden_layers = list(  # type: ignore
            filter(
                lambda l: not isinstance(
                    l[1], (self.model.__class__, torch.nn.Sequential)
                ),
                all_layers,
            )
        )

        batch_size = x.shape[0]
        hidden_outputs = []

        # We register forward hook on layers of interest, which just saves the flattened layers' outputs to list.
        # Then we execute forward pass and stack them in 2D tensor.
        def hook(module, module_in, module_out):
            arr = module_out.cpu().numpy()
            arr = arr.reshape((batch_size, -1))
            hidden_outputs.append(arr)

        new_hooks = []
        # Save handles of registered hooks, so we can clean them up later.
        for index, (name, layer) in enumerate(hidden_layers):
            if is_layer_of_interest(index, name):
                handle = layer.register_forward_hook(hook)
                new_hooks.append(handle)

        if len(new_hooks) == 0:
            raise ValueError("No hidden representations were selected.")

        # Execute forward pass.
        with torch.no_grad():
            self.model(torch.Tensor(x).to(device))

        # Cleanup.
        [i.remove() for i in new_hooks]
        return np.hstack(hidden_outputs)


def make_channel_first(x: np.array, channel_first=False):
    """
    Reshape batch to channel first.

    This function is not originally written by authors of MetaQuantus, but is sourced from the
    Quantus library as found at: https://github.com/understandable-machine-intelligence-lab/Quantus.

    Parameters
    ----------
    x: np.ndarray
         The input image.

    Returns
    -------
    np.ndarray
        Image in CxWxH format.
    """
    if channel_first:
        return x

    if len(np.shape(x)) == 4:
        return np.moveaxis(x, -1, -3)
    elif len(np.shape(x)) == 3:
        return np.moveaxis(x, -1, -2)
    else:
        raise ValueError(
            "Only batched 1d and 2d multi-channel input dimensions supported."
        )