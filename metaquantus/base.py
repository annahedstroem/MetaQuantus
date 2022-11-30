from typing import Union, List, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod
import warnings
import sklearn
import numpy as np

#from quantus.helpers.model.model_interface import ModelInterface
#from quantus.metrics.base import Metric, PerturbationMetric

class Analyser(ABC):
    """Implementation of base class."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(
        self,
        metric, # : Union[Metric, PerturbationMetric]
        nr_perturbations: int,
        model, # : ModelInterface
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
    ) -> Union[int, float, list, dict, None]:
        raise NotImplementedError
