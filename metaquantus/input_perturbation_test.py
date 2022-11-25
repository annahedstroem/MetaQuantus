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


class InputPerturbationTest(Analyser):
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
        xai_methods: Dict[str, dict],
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

            # Wrap the model.
            model_wrapped = utils.get_wrapped_model(
                model=model,
                channel_first=channel_first,
                softmax=softmax,
                device=device,
                model_predict_kwargs=model_predict_kwargs,
            )

            # Make predictions with perturbed input.
            y_preds_perturbed[p] = np.argmax(
                model_wrapped.predict(torch.Tensor(x_batch_perturbed)),
                axis=1,
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

            for x, (method, explain_func_kwargs) in enumerate(xai_methods.items()):

                # Generate explanations based on predictions.
                a_batch_preds = generate_explanations(
                    model=model,
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

                # Evaluate explanations with perturbed input.
                scores[method][p] = metric(
                    model=model,
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

                # Collect garbage.
                gc.collect()
                torch.cuda.empty_cache()

        print(y_preds_perturbed)

        return (scores, y_preds_perturbed, indices_perturbed)
