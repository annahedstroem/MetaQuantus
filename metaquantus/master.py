from typing import Callable, List, Union, Dict, Tuple, Optional, Any
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

from .utils import (
    generate_explanations,
    compute_iac_score,
    compute_iec_score,
    dump_obj,
)
from .base import Analyser
from .sanity_checks import sanity_analysis, sanity_analysis_under_perturbation


class MasterAnalyser:
    def __init__(
        self,
        analyser_suite: Dict[str, Analyser],
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
        self.analyser_suite = analyser_suite
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

        # Init empty data holders for results.
        self.results_eval_scores = {k: {} for k in self.analyser_suite}
        self.results_eval_scores_perturbed = {k: {} for k in self.analyser_suite}
        self.results_y_preds_perturbed = {k: {} for k in self.analyser_suite}
        self.results_indices_perturbed = {k: {} for k in self.analyser_suite}
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
        if any("Resilience" in test for test in list(self.analyser_suite)) and any(
            "Adversary" in test for test in list(self.analyser_suite)
        ):
            self.run_meta_consistency_analysis()

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
                fname=f"{self.today}_{self.fname}consistency_scores_{self.uid}",
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
        self.results_eval_scores = {k: {} for k in self.analyser_suite}
        self.results_eval_scores_perturbed = {k: {} for k in self.analyser_suite}
        self.results_y_preds_perturbed = {k: {} for k in self.analyser_suite}
        self.results_indices_perturbed = {k: {} for k in self.analyser_suite}
        self.results_y_true = y_batch
        self.results_y_preds = np.zeros_like(y_batch)
        self.results_indices_correct = np.zeros_like(y_batch)
        self.results_intra_scores = {}
        self.results_inter_scores = {}
        self.results_meta_consistency_scores = {}
        self.results_consistency_scores = {}

        for a, (test_name, test) in enumerate(self.analyser_suite.items()):

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

                self.results_eval_scores[test_name][method] = np.array(
                    scores
                ).astype(float)

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
                        metric=estimator,
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

        self.results_intra_scores = {k: {} for k in self.analyser_suite}

        for a, (test_name, test) in enumerate(self.analyser_suite.items()):
            for x, method in enumerate(self.xai_methods.keys()):
                self.results_intra_scores[test_name][method] = []

                for i in range(self.iterations):

                    p_values = []

                    for p in range(self.nr_perturbations):

                        p_value = compute_iac_score(
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

        for a, (test_name, test) in enumerate(self.analyser_suite.items()):

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
                    iec_score = compute_iec_score(
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
        perturbation_types = np.unique([k.split(" ")[0] for k in self.analyser_suite])

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
            meta_consistency_scores = {
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

            # Produce the results.
            shape = (self.iterations, 4)
            self.results_meta_consistency_scores[perturbation_type] = {
                "meta_consistency_scores": meta_consistency_scores,
                "MC_means": np.array(list(meta_consistency_scores.values()))
                .reshape(shape)
                .mean(axis=1),
                "MC_mean": np.array(list(meta_consistency_scores.values()))
                .reshape(shape)
                .mean(),
                "MC_std": np.array(list(meta_consistency_scores.values()))
                .reshape(shape)
                .mean(axis=1)
                .std(),
            }
            print(
                f"\n\n{perturbation_type} Perturbation Test ---> MC score="
                f"{self.results_meta_consistency_scores[perturbation_type]['MC_mean']:.2f} "
                f"({self.results_meta_consistency_scores[perturbation_type]['MC_std']:.2f})"
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
                for mc_metric, result in mc_results.items():
                    if isinstance(result, dict):
                        print(f"{mc_metric}:")
                        for k, v in result.items():
                            print(f"\t{k}: {v}")
                    else:
                        print(f"{mc_metric}: {result}")
            print("")
