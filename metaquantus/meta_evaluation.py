"""This module contains the main implementation for the meta-evaluation framework."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

from typing import Callable, Union, Dict, Optional, Any
import numpy as np
import scipy
import torch
import gc
import uuid
from datetime import datetime
from tqdm.auto import tqdm

from quantus.helpers import utils
from quantus import explain
from quantus.metrics.base import Metric, PerturbationMetric

from .helpers.utils import (
    generate_explanations,
    dump_obj,
)
from .perturbation_tests.base import PerturbationTestBase
from .helpers.sanity_checks import sanity_analysis, sanity_analysis_under_perturbation


class MetaEvaluation:
    def __init__(
        self,
        test_suite: Dict[str, PerturbationTestBase],
        xai_methods: Dict[str, Any],
        iterations: int = 5,
        nr_perturbations: int = 10,
        explain_func: Callable = explain,
        intra_measure: Callable = scipy.stats.wilcoxon,
        path: str = "/content/drive/MyDrive/Projects/MetaQuantus/results/",
        fname: str = "",
        write_to_file: bool = True,
        uid: Optional[str] = None,
        sanity_check: Optional[str] = None,
    ):
        """
        This class implements the Meta-evaluation Framework.

        Parameters
        ----------
        test_suite: dict
            A dictionary of tests, IPT and MPT test.
        xai_methods: dict
            A dictionary of XAI methods that are supported in Quantus.
        iterations: int
            The number of iterations to run the experiment.
        nr_perturbations: int
            The number of perturbations.
        explain_func: callable
            The function used for creating the explanation.
        intra_measure: callable
            The statistical significane measure used.
        path: str
            The path for saving the plot.
        fname: str
            The filename.
        write_to_file: boolean
            Indicates if writing to file.
        uid: str
            A unique identifier.
        sanity_check: boolean
            Indicates whether to sanity exercise is performed.

        Returns
        -------
        None
        """
        self.test_suite = test_suite
        self.xai_methods = xai_methods
        self.path = path
        self.fname = fname
        self.iterations = iterations
        self.write_to_file = write_to_file
        self.nr_perturbations = nr_perturbations
        self.explain_func = explain_func
        self.intra_measure = intra_measure
        self.uid = uid
        self.sanity_check = sanity_check
        self.debug = debug

        # Init empty data holders for results.
        self.results_eval_scores = {k: {} for k in self.test_suite}
        self.results_eval_scores_perturbed = {k: {} for k in self.test_suite}
        self.results_y_preds_perturbed = {k: {} for k in self.test_suite}
        self.results_indices_perturbed = {k: {} for k in self.test_suite}
        self.results_y_true = None
        self.results_y_preds = None
        self.results_indices_correct = None
        self.results_intra_scores = {}
        self.results_inter_scores = {}
        self.results_meta_consistency_scores = {}
        self.results_consistency_scores = {}

        assert isinstance(self.xai_methods, dict), (
            f"The XAI methods 'xai_methods' i.e., {self.xai_methods} must "
            f"be a dict where the key is the explanation method and the values"
            f" are the explain_func_kwargs (dict), model_predict_kwargs "
            f"(dict) and softmax (bool)."
        )

        if self.uid is None:
            self.uid = str(uuid.uuid4())[:4]
        self.today = datetime.today().strftime("%d%m%Y")

        if self.sanity_check:
            assert self.sanity_check in [
                "Estimator_Different",
                "Estimator_Same",
            ], "Sanity checks can be None, 'Estimator_Different' or 'Estimator_Same'."

    def __call__(
        self,
        estimator: Union[Metric, PerturbationMetric],
        model: torch.nn.Module,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None] = None,
        s_batch: Union[np.array, None] = None,
        channel_first: Optional[bool] = True,
        softmax: Optional[bool] = False,
        device: Optional[str] = None,
        model_predict_kwargs: Optional[Dict[str, Any]] = {},
        lower_is_better: bool = False,
        reverse_scoring: bool = True,
    ):
        """
        Running meta-evalaution.

        Parameters
        ----------
        estimator: metric, perturbationmetric
            The estimator to run the test on.
        model: torch.nn
            The model used in evaluation.
        x_batch: np.array
            The input data.
        y_batch: np.array
            The labels.
        a_batch: np.array
            The explantions.
        s_batch: np.array
            The segmentation masks.
        channel_first: bool
            Indicates if channels is first.
        softmax: bool
            Indicates if the softmax (or logits) are used.
        device: torch.device
            The device used, to enable GPUs.
        model_predict_kwargs: dict
            A dictionary with predict kwargs for the model.
        lower_is_better: boolean
            Indicates if lower values are better for the estimators, e.g., True for the Robustness category.
        reverse_scoring: boolean
            Indicates if reserve scoring should eb applied.

        Returns
        -------
        self
        """
        print(f"UID={self.uid}")

        # Make perturbation.
        self.run_perturbation_analysis(
            estimator=estimator,
            model=model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            s_batch=s_batch,
            channel_first=channel_first,
            softmax=softmax,
            device=device,
        )

        # Run inference.
        self.run_intra_analysis(reverse_scoring=reverse_scoring)
        self.run_inter_analysis(lower_is_better=lower_is_better)

        # Check that both test parts exist in the test suite.
        if any("Resilience" in test for test in list(self.test_suite)) and any(
            "Adversary" in test for test in list(self.test_suite)
        ):
            self.run_meta_consistency_analysis()
            self.print_meta_consistency_scores()

        if self.write_to_file:
            dump_obj(
                obj=self.results_eval_scores,
                path=self.path,
                fname=f"{self.today}_{self.fname}_eval_scores_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.results_eval_scores_perturbed,
                path=self.path,
                fname=f"{self.today}_{self.fname}_eval_scores_perturbed_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.results_indices_perturbed,
                path=self.path,
                fname=f"{self.today}_{self.fname}_indices_perturbed_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.results_indices_correct,
                path=self.path,
                fname=f"{self.today}_{self.fname}_indices_correct_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.results_intra_scores,
                path=self.path,
                fname=f"{self.today}_{self.fname}_intra_scores_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.results_inter_scores,
                path=self.path,
                fname=f"{self.today}_{self.fname}_inter_scores_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.results_meta_consistency_scores,
                path=self.path,
                fname=f"{self.today}_{self.fname}_meta_consistency_scores_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.results_consistency_scores,
                path=self.path,
                fname=f"{self.today}_{self.fname}_consistency_scores_{self.uid}",
                use_json=True,
            )

        return self

    def run_perturbation_analysis(
        self,
        estimator: Union[Metric, PerturbationMetric],
        model: torch.nn.Module,
        x_batch: np.array,
        y_batch: np.array,
        a_batch: Union[np.array, None] = None,
        s_batch: Union[np.array, None] = None,
        channel_first: Optional[bool] = True,
        softmax: Optional[bool] = False,
        device: Optional[str] = None,
        model_predict_kwargs: Optional[Dict[str, Any]] = {},
    ):

        # It is important to wipe out earlier results of the same master init when benchmarking.
        self.results_eval_scores = {k: {} for k in self.test_suite}
        self.results_eval_scores_perturbed = {k: {} for k in self.test_suite}
        self.results_y_preds_perturbed = {k: {} for k in self.test_suite}
        self.results_indices_perturbed = {k: {} for k in self.test_suite}
        self.results_y_true = y_batch
        self.results_y_preds = np.zeros_like(y_batch)
        self.results_indices_correct = np.zeros_like(y_batch)
        self.results_intra_scores = {}
        self.results_inter_scores = {}
        self.results_meta_consistency_scores = {}
        self.results_consistency_scores = {}

        for a, (test_name, test) in enumerate(self.test_suite.items()):

            self.results_eval_scores[test_name] = {}
            self.results_eval_scores_perturbed[test_name] = {}
            self.results_y_preds_perturbed[test_name] = {}
            self.results_indices_perturbed[test_name] = {}

            if not any(x in test_name for x in ["Adversary", "Resilience"]):
                raise ValueError(
                    "Either 'Adversary' or 'Resilience' must be in test name."
                )
            print(test_name)

            # This is needed for iterator (zipped over x_batch, y_batch, a_batch, s_batch).
            if s_batch is None:
                s_batch = [None for _ in x_batch]

            # Wrap the model and put in evaluation mode.
            model.to(device)
            model.eval()
            model_wrapped = utils.get_wrapped_model(
                model=model,
                channel_first=channel_first,
                softmax=softmax,
                device=device,
                model_predict_kwargs=model_predict_kwargs,
            )

            # Make predictions with unperturbed model on unperturbed input.
            self.results_y_preds = np.argmax(
                model_wrapped.predict(torch.Tensor(x_batch)),
                axis=1,
            ).astype(int)

            self.results_indices_correct = self.results_y_true == self.results_y_preds

            # Loop over all explanation methods and save scores with no perturbation applied.
            for x, (method, explain_func_kwargs) in enumerate(self.xai_methods.items()):

                if self.sanity_check is not None:
                    scores = sanity_analysis(
                        sanity_type=self.sanity_check, items=len(x_batch)
                    )

                else:

                    # Generate explanations based on predictions.
                    a_batch_preds = generate_explanations(
                        model=model,
                        x_batch=x_batch,
                        y_batch=self.results_y_preds,
                        explain_func=self.explain_func,
                        explain_func_kwargs={
                            **explain_func_kwargs,
                            **{"method": method},
                        },
                        abs=estimator.abs,
                        normalise=estimator.normalise,
                        normalise_func=estimator.normalise_func,
                        normalise_func_kwargs=estimator.normalise_func_kwargs,
                        device=device,
                    )

                    scores = estimator(
                        model=model,
                        x_batch=x_batch,
                        y_batch=self.results_y_preds,
                        a_batch=a_batch_preds,
                        s_batch=s_batch,
                        channel_first=channel_first,
                        explain_func=self.explain_func,
                        explain_func_kwargs=explain_func_kwargs,
                        model_predict_kwargs=model_predict_kwargs,
                        softmax=softmax,
                        device=device,
                    )

                self.results_eval_scores[test_name][method] = np.array(scores).astype(
                    float
                )

            # Loop over all iterations and save scores with perturbation applied.
            # All explanation methods will be iterated within the metric.

            for i in tqdm(range(self.iterations), desc="Iterations"):

                # Run sanity checks.
                if self.sanity_check is not None:

                    # Run test!
                    (
                        scores_perturbed,
                        y_preds_perturbed,
                        indices_perturbed,
                    ) = sanity_analysis_under_perturbation(
                        sanity_type=self.sanity_check,
                        items=len(x_batch),
                        nr_perturbations=self.nr_perturbations,
                        xai_methods=self.xai_methods,
                        unperturbed_scores=self.results_eval_scores[test_name],
                    )

                else:

                    # Run test!
                    scores_perturbed, y_preds_perturbed, indices_perturbed = test(
                        estimator=estimator,
                        nr_perturbations=self.nr_perturbations,
                        xai_methods=self.xai_methods,
                        model=model,
                        x_batch=x_batch,
                        y_batch=self.results_y_preds,
                        a_batch=None,
                        s_batch=s_batch,
                        channel_first=channel_first,
                        explain_func=self.explain_func,
                        model_predict_kwargs=model_predict_kwargs,
                        softmax=softmax,
                        device=device,
                    )

                self.results_eval_scores_perturbed[test_name][i] = scores_perturbed
                self.results_y_preds_perturbed[test_name][i] = y_preds_perturbed
                self.results_indices_perturbed[test_name][i] = indices_perturbed

                # Collect garbage.
                gc.collect()
                torch.cuda.empty_cache()

    def run_intra_analysis(self, reverse_scoring: bool = True) -> Dict:
        """Make IAC inference after perturbing inputs to the evaluation problem and then storing scores."""

        self.results_intra_scores = {k: {} for k in self.test_suite}

        for a, (test_name, test) in enumerate(self.test_suite.items()):
            for x, method in enumerate(self.xai_methods.keys()):
                self.results_intra_scores[test_name][method] = []

                for i in range(self.iterations):

                    p_values = []

                    for p in range(self.nr_perturbations):

                        p_value = self.compute_iac_score(
                            q=self.results_eval_scores[test_name][method],
                            q_hat=self.results_eval_scores_perturbed[test_name][i][
                                method
                            ][p],
                            indices=self.results_indices_perturbed[test_name][i][p],
                            test_name=test_name,
                            measure=self.intra_measure,
                            reverse_scoring=reverse_scoring,
                        )
                        p_values.append(p_value)

                    self.results_intra_scores[test_name][method].append(
                        np.array(p_values)
                    )

                self.results_intra_scores[test_name][method] = np.array(
                    self.results_intra_scores[test_name][method]
                )

        return self.results_intra_scores

    def run_inter_analysis(self, lower_is_better: bool):
        """
        Make IEC inference after perturbing inputs to the evaluation problem and then storing scores.

        Parameters
        ----------
        lower_is_better

        Returns
        -------

        """

        self.results_inter_scores = {}
        shape = (self.iterations, self.nr_perturbations)

        for a, (test_name, test) in enumerate(self.test_suite.items()):

            self.results_inter_scores[test_name] = []

            for i in range(self.iterations):

                # Create placeholders for unperturbed (Q_star) and perturbed scores (Q_hat).
                Q_star = np.zeros((len(self.xai_methods), len(self.results_y_preds)))
                Q_hat = np.zeros((len(self.xai_methods), len(self.results_y_preds)))

                # Save unperturbed scores in Q_star.
                for x, (method, explain_func_kwargs) in enumerate(
                    self.xai_methods.items()
                ):
                    Q_star[x] = self.results_eval_scores[test_name][method]

                # Save perturbed scores in Q_hat.
                for k in range(self.nr_perturbations):

                    for x, (method, explain_func_kwargs) in enumerate(
                        self.xai_methods.items()
                    ):
                        Q_hat[x] = self.results_eval_scores_perturbed[test_name][i][
                            method
                        ][k]

                    # Different conditions for calculating IEC depending on perturbation type.
                    iec_score = self.compute_iec_score(
                        Q_star=Q_star,
                        Q_hat=Q_hat,
                        indices=self.results_indices_perturbed[test_name][i][k],
                        lower_is_better=lower_is_better,
                        test_name=test_name,
                    )

                    self.results_inter_scores[test_name].append(iec_score)

            self.results_inter_scores[test_name] = np.array(
                self.results_inter_scores[test_name]
            ).reshape(shape)

        return self.results_inter_scores

    def run_meta_consistency_analysis(self) -> dict:
        """
        Compute the meta consistency (MC score)

        Assumes tests are called one of the following:
        - 'Model Resilience Test'
        - 'Model Adversary Test'
        - 'Input Resilience Test'
        - 'Input Adversary Test'

        Returns
        -------
        dict
            The meta-consistency results.
        """
        perturbation_types = np.unique([k.split(" ")[0] for k in self.test_suite])

        for perturbation_type in perturbation_types:

            assert perturbation_type in [
                "Model",
                "Input",
            ], "The 'perturbation_type' needs to either be 'Model' or 'Input'."

            self.results_consistency_scores[perturbation_type] = {}

            # Get intra scores of all explanation methods.
            shape = (len(self.xai_methods), self.iterations, self.nr_perturbations)
            self.results_consistency_scores[perturbation_type]["intra_scores_res"] = (
                np.array(
                    list(
                        self.results_intra_scores[
                            f"{perturbation_type} Resilience Test"
                        ].values()
                    )
                )
                .flatten()
                .reshape(shape)
            )
            self.results_consistency_scores[perturbation_type]["intra_scores_adv"] = (
                np.array(
                    list(
                        self.results_intra_scores[
                            f"{perturbation_type} Adversary Test"
                        ].values()
                    )
                )
                .flatten()
                .reshape(shape)
            )

            # Get inter scores of all explanation methods.
            shape = (self.iterations, self.nr_perturbations)
            self.results_consistency_scores[perturbation_type][
                "inter_scores_res"
            ] = np.array(
                self.results_inter_scores[f"{perturbation_type} Resilience Test"]
            ).reshape(
                shape
            )
            self.results_consistency_scores[perturbation_type][
                "inter_scores_adv"
            ] = np.array(
                self.results_inter_scores[f"{perturbation_type} Adversary Test"]
            ).reshape(
                shape
            )

            # Get the mean scores, over the right axes.
            consistency_scores = {
                "IAC_{NR}": self.results_consistency_scores[perturbation_type][
                    "intra_scores_res"
                ].mean(axis=(0, 2)),
                "IAC_{AR}": self.results_consistency_scores[perturbation_type][
                    "intra_scores_adv"
                ].mean(axis=(0, 2)),
                "IEC_{NR}": self.results_consistency_scores[perturbation_type][
                    "inter_scores_res"
                ].mean(axis=1),
                "IEC_{AR}": self.results_consistency_scores[perturbation_type][
                    "inter_scores_adv"
                ].mean(axis=1),
            }

            # Compute the results.
            consistency_results = {
                "IAC_{NR} mean": consistency_scores["IAC_{NR}"].mean(),
                "IAC_{NR} std": consistency_scores["IAC_{NR}"].std(),
                "IAC_{AR} mean": consistency_scores["IAC_{AR}"].mean(),
                "IAC_{AR} std": consistency_scores["IAC_{NR}"].std(),
                "IEC_{NR} mean": consistency_scores["IEC_{NR}"].mean(),
                "IEC_{NR} std": consistency_scores["IEC_{NR}"].std(),
                "IEC_{AR} mean": consistency_scores["IEC_{AR}"].mean(),
                "IEC_{AR} std": consistency_scores["IEC_{AR}"].std(),
            }

            # Produce the results.
            shape = (4, self.iterations)
            self.results_meta_consistency_scores[perturbation_type] = {
                "consistency_scores": consistency_scores,
                "consistency_results": consistency_results,
                "MC_means": np.array(list(consistency_scores.values()))
                .reshape(shape)
                .mean(axis=0),
                "MC_mean": np.array(list(consistency_scores.values()))
                .reshape(shape)
                .mean(),
                "MC_std": np.array(list(consistency_scores.values()))
                .reshape(shape)
                .mean(axis=0)
                .std(),
            }
            print(
                f"\n{perturbation_type} Perturbation Test ---> MC score="
                f"{self.results_meta_consistency_scores[perturbation_type]['MC_mean']:.4f} "
                f"({self.results_meta_consistency_scores[perturbation_type]['MC_std']:.4f})"
            )

        return self.results_meta_consistency_scores

    def print_meta_consistency_scores(self) -> None:
        """Print MC scores in a human-readable way."""
        if self.results_meta_consistency_scores:
            print("")
            for (
                perturbation_type,
                mc_results,
            ) in self.results_meta_consistency_scores.items():
                print(f"{perturbation_type} Perturbation Test")
                for mc_metric, result in mc_results.items():
                    if isinstance(result, dict):
                        print(f"\t{mc_metric}:")
                        for k, v in result.items():
                            print(f"\t\t{k}: {v}")
                    else:
                        print(f"\t{mc_metric}: {result}")
            print("")

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

    """Getters."""

    def get_results_eval_scores(self):
        return self.results_eval_scores

    def get_results_eval_scores_perturbed(self):
        return self.results_eval_scores_perturbed

    def get_results_y_preds_perturbed(self):
        return self.results_y_preds_perturbed

    def get_results_indices_perturbed(self):
        return self.results_indices_perturbed

    def get_results_y_true(self):
        return self.results_y_true

    def get_results_y_preds(self):
        return self.results_y_preds

    def get_results_indices_correct(self):
        return self.results_indices_correct

    def get_results_intra_scores(self):
        return self.results_intra_scores

    def get_results_inter_scores(self):
        return self.results_inter_scores

    def get_results_meta_consistency_scores(self):
        return self.results_meta_consistency_scores

    def get_results_consistency_scores(self):
        return self.results_consistency_scores
