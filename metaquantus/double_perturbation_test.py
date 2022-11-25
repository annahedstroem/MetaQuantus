from typing import Union, Dict, List, Any, Optional, Callable, Tuple
import copy
import gc
import torch
import numpy as np

from quantus.helpers import utils
from quantus.helpers.model.model_interface import ModelInterface
from quantus.metrics.base import Metric, PerturbationMetric


from .base import Analyser
from .utils import generate_explanations


class DoublePerturbationTest(Analyser):
    def __init__(
        self,
        type: str,
        noise: float,
    ):
        super().__init__()
        self.noise = noise
        self.type = type.lower()

        assert self.noise != 0.0, "Model noise ('std') cannot be zero."

    def __call__(
        self,
        metric: Union[Metric, PerturbationMetric],
        nr_perturbations: int,
        model: ModelInterface,
        x_batch: np.ndarray,
        y_batch: Optional[np.ndarray],
        a_batch: Optional[np.ndarray],
        s_batch: Optional[np.ndarray],
        channel_first: Optional[bool],
        explain_func: Optional[Callable],
        explain_func_kwargs: Optional[Dict[str, Any]],
        model_predict_kwargs: Optional[Dict],
        softmax: Optional[bool],
        device: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Determine shape of results.
        scores = np.ndarray((nr_perturbations, len(x_batch)), dtype=float)
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

            # Perturb the input.
            x_batch_perturbed = copy.copy(x_batch)
            x_batch_perturbed += np.random.uniform(
                low=-self.noise,
                high=self.noise,
                size=x_batch_perturbed.shape,
            )

            # Clip the input so that it is within same domain.
            x_batch_perturbed = np.clip(
                x_batch_perturbed, x_batch.min(), x_batch_perturbed.max()
            )

            # Make predictions with perturbed input.
            y_preds_perturbed[p] = np.argmax(
                model_perturbed.predict(torch.Tensor(x_batch_perturbed)),
                axis=1,
            )

            # Generate explanations based on predictions.
            a_batch_preds = generate_explanations(
                model=model_perturbed.get_model(),
                x_batch=x_batch_perturbed,
                y_batch=y_batch,
                explain_func=explain_func,
                explain_func_kwargs=explain_func_kwargs,
                abs=metric.abs,
                normalise=metric.normalise,
                normalise_func=metric.normalise_func,
                normalise_func_kwargs=metric.normalise_func_kwargs,
                device=device,
            )

            # Evaluate explanations with perturbed input.
            scores[p] = metric(
                model=model_perturbed.get_model(),
                x_batch=x_batch_perturbed,
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

            # Save indices based on perturbation type.
            if self.type == "resilience":
                indices_perturbed[p] = y_batch == y_preds_perturbed[p]
            elif self.type == "adversary":
                indices_perturbed[p] = y_batch != y_preds_perturbed[p]
            else:
                raise ValueError(
                    "The perturbation type must either be 'Resilience' or 'Adversary'."
                )

            # Collect garbage.
            gc.collect()
            torch.cuda.empty_cache()

        return scores.astype(float), y_preds_perturbed, indices_perturbed
