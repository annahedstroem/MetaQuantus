"""This module contains the implementation for the Model Perturbation Test."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

from typing import Union, Dict, List, Any, Optional, Callable, Tuple
import gc
import torch
import numpy as np

from quantus.helpers import utils
from quantus.helpers.model.model_interface import ModelInterface
from quantus.metrics.base import Metric, PerturbationMetric

from .base import PerturbationTestBase
from MetaQuantus.metaquantus.utils import generate_explanations


class ModelPerturbationTest(PerturbationTestBase):
    def __init__(
        self,
        type: str,
        noise_type: str,
        mean: float,
        std: float,
    ):
        """
        Implementation of Input Perturbation Test.

        Parameters
        ----------

        noise: float
            Noise type
        Parameters
        ----------
        type: str
            The space which the perturbation is applied: either 'resilience' or 'adversary'.
        noise_type: str
            The type of noise applied: either 'multiplicative' or 'additive'.
        mean: float
            The mean of the normal distribution.
        std: float
            The mean of the normal distribution.
        """
        super().__init__()
        self.type = type.lower()
        self.noise_type = noise_type
        self.mean = mean
        self.std = std

        assert self.std != 0.0, "Model std ('std') cannot be zero."

    def __call__(
        self,
        metric: Union[Metric, PerturbationMetric],
        xai_methods: Dict[str, dict],
        nr_perturbations: int,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: Optional[np.ndarray],
        channel_first: Optional[bool],
        explain_func: Optional[Callable],
        model_predict_kwargs: Optional[Dict],
        softmax: Optional[bool],
        device: Optional[str],
    ) -> Tuple[dict, np.ndarray, dict]:

        # Determine the shape of results.
        scores = {
            k: np.ndarray((nr_perturbations, len(x_batch)), dtype=float)
            for k in xai_methods
        }
        y_preds_perturbed = np.ndarray((nr_perturbations, len(x_batch)), dtype=int)
        indices_perturbed = np.ndarray((nr_perturbations, len(x_batch)), dtype=bool)

        for p in range(nr_perturbations):

            # Create a perturbed model, to re-generate explanations with.
            model_perturbed = utils.get_wrapped_model(
                model=model,
                channel_first=channel_first,
                softmax=softmax,
                device=device,
                model_predict_kwargs=model_predict_kwargs,
            )

            # Add noise to model weights.
            model_perturbed = model_perturbed.sample(
                mean=self.mean, std=self.std, noise_type=self.noise_type
            )

            # Wrap model.
            model_perturbed = utils.get_wrapped_model(
                model=model_perturbed,
                channel_first=channel_first,
                softmax=softmax,
                device=device,
                model_predict_kwargs=model_predict_kwargs,
            )

            # Make predictions with perturbed input.
            y_preds_perturbed[p] = np.argmax(
                model_perturbed.predict(torch.Tensor(x_batch)),
                axis=1,
            ).astype(int)

            # Save indices based on perturbation type.
            if self.type == "resilience":
                indices_perturbed[p] = y_batch == y_preds_perturbed[p]
            elif self.type == "adversary":
                indices_perturbed[p] = y_batch != y_preds_perturbed[p]
            else:
                raise ValueError(
                    "The perturbation type must either be 'Resilience' or 'Adversary'."
                )

            for x, (method, explain_func_kwargs) in enumerate(xai_methods.items()):

                # Generate explanations based on predictions.
                a_batch_preds = generate_explanations(
                    model=model_perturbed.get_model(),
                    x_batch=x_batch,
                    y_batch=y_batch,
                    explain_func=explain_func,
                    explain_func_kwargs={**explain_func_kwargs, **{"method": method}},
                    abs=metric.abs,
                    normalise=metric.normalise,
                    normalise_func=metric.normalise_func,
                    normalise_func_kwargs=metric.normalise_func_kwargs,
                    device=device,
                )

                # Evaluate explanations with perturbed model.
                scores[method][p] = metric(
                    model=model_perturbed.get_model(),
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch_preds,
                    s_batch=s_batch,
                    explain_func=explain_func,
                    explain_func_kwargs=explain_func_kwargs,
                    model_predict_kwargs=model_predict_kwargs,
                    channel_first=channel_first,
                    softmax=softmax,
                    device=device,
                )

                # Collect garbage.
                gc.collect()
                torch.cuda.empty_cache()

        return scores, y_preds_perturbed, indices_perturbed