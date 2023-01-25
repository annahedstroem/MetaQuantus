"""This module contains the implementation for the Perturbation Test base class."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

from typing import Union, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import scipy
import torch
import quantus

class PerturbationTestBase(ABC):
    """Implementation of base class for the PerturbationTest."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(
        self,
        estimator,
        nr_perturbations: int,
        model: torch.nn.Module,
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
