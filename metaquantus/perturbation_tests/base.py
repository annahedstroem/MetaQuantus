"""This module contains the implementation for the Perturbation Test base class."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

from typing import Union, Optional, Callable, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
import scipy

from quantus.helpers.model.model_interface import ModelInterface
from quantus.metrics.base import Metric, PerturbationMetric


class PerturbationTestBase(ABC):
    """Implementation of base class for the PerturbationTest."""

    def __init__(self):
        pass

    @abstractmethod
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
    ) -> Union[int, float, list, dict, None]:
        raise NotImplementedError

    @staticmethod
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

    def compute_iec_score(
        self,
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
            return self.compute_iec_adversary(
                Q_star=Q_star,
                Q_hat=Q_hat,
                indices=indices,
                lower_is_better=lower_is_better,
            )
        elif "Resilience" in test_name:
            return self.compute_iec_resilience(
                Q_star=Q_star, Q_hat=Q_hat, indices=indices
            )

        return np.nan

    @staticmethod
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

    @staticmethod
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
