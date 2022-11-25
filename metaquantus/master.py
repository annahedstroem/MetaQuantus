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

        # Init empty data holders.
        self.eval_scores = {}
        self.eval_scores_perturbed = {}
        self.y_true = None
        self.y_preds = None
        self.y_preds_perturbed = {}
        self.indices_perturbed = {}
        self.indices_correct = None
        self.intra_scores = {}
        self.inter_scores = {}
        self.mc_scores = {}
        self.raw_scores = {}

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
                "NR",
                "AR",
            ], "Sanity checks can be None, 'NR' or 'AR'."

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
        self.run_meta_consistency_analysis()

        if self.write_to_file:
            dump_obj(
                obj=self.eval_scores,
                path=self.path,
                fname=f"{self.today}_{self.fname}_eval_scores_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.eval_scores_perturbed,
                path=self.path,
                fname=f"{self.today}_{self.fname}_eval_scores_perturbed_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.indices_perturbed,
                path=self.path,
                fname=f"{self.today}_{self.fname}_indices_perturbed_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.indices_correct,
                path=self.path,
                fname=f"{self.today}_{self.fname}_indices_correct_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.intra_scores,
                path=self.path,
                fname=f"{self.today}_{self.fname}_intra_scores_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.inter_scores,
                path=self.path,
                fname=f"{self.today}_{self.fname}_inter_scores_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.mc_scores,
                path=self.path,
                fname=f"{self.today}_{self.fname}_mc_scores_{self.uid}",
                use_json=True,
            )
            dump_obj(
                obj=self.raw_scores,
                path=self.path,
                fname=f"{self.today}_{self.fname}_raw_scores_{self.uid}",
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

        # Important to wipe out earlier results of the same master intialisation.
        self.eval_scores = {k: {} for k in self.analyser_suite}
        self.eval_scores_perturbed = {k: {} for k in self.analyser_suite}
        self.y_preds_perturbed = {k: {} for k in self.analyser_suite}
        self.indices_perturbed = {k: {} for k in self.analyser_suite}
        self.y_true = y_batch
        self.y_preds = np.zeros_like(y_batch)
        self.indices_correct = np.zeros_like(y_batch)
        self.intra_scores = {}
        self.inter_scores = {}
        self.mc_scores = {}
        self.raw_scores = {}

        for a, (analyser_name, analyser) in enumerate(self.analyser_suite.items()):

            self.eval_scores[analyser_name] = {}
            self.eval_scores_perturbed[analyser_name] = {}
            self.y_preds_perturbed[analyser_name] = {}
            self.indices_perturbed[analyser_name] = {}

            if not any(x in analyser_name for x in ["Adversary", "Resilience"]):
                raise ValueError(
                    "Either 'Adversary' or 'Resilience' must be in analyser name."
                )
            print(analyser_name)

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
            self.y_preds = np.argmax(
                model_wrapped.predict(torch.Tensor(x_batch)),
                axis=1,
            ).astype(int)

            self.indices_correct = self.y_true == self.y_preds

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
                        y_batch=self.y_preds,
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
                        y_batch=self.y_preds,
                        a_batch=a_batch_preds,
                        s_batch=s_batch,
                        channel_first=channel_first,
                        explain_func=self.explain_func,
                        explain_func_kwargs=explain_func_kwargs,
                        model_predict_kwargs=model_predict_kwargs,
                        softmax=softmax,
                        device=device,
                    )

                self.eval_scores[analyser_name][method] = np.array(scores).astype(float)

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
                        unperturbed_scores=self.eval_scores[analyser_name],
                    )

                else:

                    # Run test!
                    scores_perturbed, y_preds_perturbed, indices_perturbed = analyser(
                        metric=estimator,
                        nr_perturbations=self.nr_perturbations,
                        xai_methods=self.xai_methods,
                        model=model,
                        x_batch=x_batch,
                        y_batch=self.y_preds,
                        a_batch=None,
                        s_batch=s_batch,
                        channel_first=channel_first,
                        explain_func=self.explain_func,
                        model_predict_kwargs=model_predict_kwargs,
                        softmax=softmax,
                        device=device,
                    )

                self.eval_scores_perturbed[analyser_name][i] = scores_perturbed
                self.y_preds_perturbed[analyser_name][i] = y_preds_perturbed
                self.indices_perturbed[analyser_name][i] = indices_perturbed

                # Collect garbage.
                gc.collect()
                torch.cuda.empty_cache()

    def run_intra_analysis(self, reverse_scoring: bool = True) -> Dict:
        """Make IAC inference after perturbing inputs to the evaluation problem and then storing scores."""

        self.intra_scores = {k: {} for k in self.analyser_suite}

        for a, (analyser_name, analyser) in enumerate(self.analyser_suite.items()):
            for x, method in enumerate(self.xai_methods.keys()):
                self.intra_scores[analyser_name][method] = []

                for i in range(self.iterations):

                    p_values = []

                    for p in range(self.nr_perturbations):

                        p_value = compute_iac_score(
                            q=self.eval_scores[analyser_name][method],
                            q_hat=self.eval_scores_perturbed[analyser_name][i][method][
                                p
                            ],
                            indices=self.indices_perturbed[analyser_name][i][p],
                            analyser_name=analyser_name,
                            measure=self.intra_measure,
                            reverse_scoring=reverse_scoring,
                        )
                        p_values.append(p_value)

                    self.intra_scores[analyser_name][method].append(np.array(p_values))

                self.intra_scores[analyser_name][method] = np.array(
                    self.intra_scores[analyser_name][method]
                )

        return self.intra_scores

    def run_inter_analysis(self, lower_is_better: bool):
        """
        Make IEC inference after perturbing inputs to the evaluation problem and then storing scores.

        Parameters
        ----------
        lower_is_better

        Returns
        -------

        """

        self.inter_scores = {}

        for a, (analyser_name, analyser) in enumerate(self.analyser_suite.items()):

            self.inter_scores[analyser_name] = []

            for i in range(self.iterations):

                # Create placeholders for unperturbed (Q_star) and perturbed scores (Q_hat).
                Q_star = np.zeros((len(self.xai_methods), len(self.y_preds)))
                Q_hat = np.zeros((len(self.xai_methods), len(self.y_preds)))

                # Save unperturbed scores in Q_star.
                for x, (method, explain_func_kwargs) in enumerate(
                    self.xai_methods.items()
                ):
                    Q_star[x] = self.eval_scores[analyser_name][method]

                # Save perturbed scores in Q_hat.
                for k in range(self.nr_perturbations):

                    for x, (method, explain_func_kwargs) in enumerate(
                        self.xai_methods.items()
                    ):
                        Q_hat[x] = self.eval_scores_perturbed[analyser_name][i][method][
                            k
                        ]

                    # Different conditions for calculating IEC depending on perturbation type.
                    iec_score = compute_iec_score(
                        Q_star=Q_star,
                        Q_hat=Q_hat,
                        indices=self.indices_perturbed[analyser_name][i][k],
                        lower_is_better=lower_is_better,
                        analyser_name=analyser_name,
                    )

                    self.inter_scores[analyser_name].append(iec_score)

            self.inter_scores[analyser_name] = np.array(
                self.inter_scores[analyser_name]
            )

        return self.inter_scores

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
            ], "The 'perturbation_type' is either 'Model' or 'Input'."

            self.raw_scores[perturbation_type] = {}

            # Get intra scores of all explanation methods.
            shape = (len(self.xai_methods), self.iterations, self.nr_perturbations)
            self.raw_scores[perturbation_type]["intra_scores_res"] = (
                np.array(
                    list(
                        self.intra_scores[
                            f"{perturbation_type} Resilience Test"
                        ].values()
                    )
                )
                .flatten()
                .reshape(shape)
            )
            self.raw_scores[perturbation_type]["intra_scores_adv"] = (
                np.array(
                    list(
                        self.intra_scores[
                            f"{perturbation_type} Adversary Test"
                        ].values()
                    )
                )
                .flatten()
                .reshape(shape)
            )

            # Get inter scores of all explanation methods.
            shape = (self.iterations, self.nr_perturbations)
            self.raw_scores[perturbation_type]["inter_scores_res"] = np.array(
                self.inter_scores[f"{perturbation_type} Resilience Test"]
            ).reshape(shape)
            self.raw_scores[perturbation_type]["inter_scores_adv"] = np.array(
                self.inter_scores[f"{perturbation_type} Adversary Test"]
            ).reshape(shape)

            # Get the mean scores, over the right axes.
            mc_scores = {
                    "IAC_{NR}": self.raw_scores[perturbation_type]["intra_scores_res"].mean(axis=(0, 2)),
                    "IAC_{AR}": self.raw_scores[perturbation_type]["intra_scores_adv"].mean(axis=(0, 2)),
                    "IEC_{NR}": self.raw_scores[perturbation_type]["inter_scores_res"].mean(axis=1),
                    "IEC_{AR}": self.raw_scores[perturbation_type]["inter_scores_adv"].mean(axis=1),
            }

            # Produce the results.
            shape = (self.iterations, 4)
            self.mc_scores[perturbation_type] = {
                "MC_scores": mc_scores,
                "MC_means": np.array(list(mc_scores.values())).reshape(shape).mean(axis=1),
                "MC_mean": np.array(list(mc_scores.values())).reshape(shape).mean(),
                "MC_std": np.array(list(mc_scores.values())).reshape(shape).mean(axis=1).std(),
            }
            print(
                f"\n\n{perturbation_type} Perturbation Test ---> MC score="
                f"{self.mc_scores[perturbation_type]['MC_mean']:.2f} "
                f"({self.mc_scores[perturbation_type]['MC_std']:.2f})"
            )

        return self.mc_scores

    def get_mc_scores(self) -> None:
        """Print MC scores in a human-readable way."""
        for perturbation_type, mc_results in self.mc_scores.items():
            for mc_metric, result in mc_results.items():
                if isinstance(result, dict):
                    print(f"{mc_metric}:")
                    for k, v in result.items():
                        print(f"\t{k}: {v}")
                else:
                    print(f"{mc_metric}: {result:.2f}")
