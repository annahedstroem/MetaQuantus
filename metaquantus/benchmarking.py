"""This module contains the implementation for runnning benchmarking of the Meta Evaluation class."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

from typing import Dict, Any, Tuple
import torch
import gc
import uuid
from datetime import datetime
from quantus.metrics.base import Metric, PerturbationMetric

from .meta_evaluation import MetaEvaluation
from .helpers.utils import *


class MetaEvaluationBenchmarking:
    def __init__(
        self,
        master: MetaEvaluation,
        estimators: Dict[
            str, Dict[str, Tuple[Union[Metric, PerturbationMetric], bool]]
        ],
        experimental_settings: Dict[str, Dict[str, Any]],
        path: str = "/content/drive/MyDrive/Projects/MetaQuantus/results/",
        folder: str = "benchmarks/",
        write_to_file: bool = True,
        keep_results: bool = False,
        channel_first: Optional[bool] = True,
        softmax: Optional[bool] = False,
        device: Optional[str] = None,
    ):
        """
        This class implements the main logic to conduct benchmarking.

        Parameters
        ----------
        master: metaevaluation
            An intialised MetaEvalaution object.
        estimators: dict
            A dictionary of the estimators to benchmark.
        experimental_settings: dict
            A dictionary of the experimental settings including model, data, label etc.
        path: str
            The path for saving the plot.
        folder: str
            The folder name.
        write_to_file: boolean
            Indicates if writing to file.
        keep_results boolean
            Indicates if saving results.
        channel_first: bool
            Indicates if channels is first.
        softmax: bool
            Indicates if the softmax (or logits) are used.
        device: torch.device
            The device used, to enable GPUs.

        Returns
        -------
        None
        """
        self.master = master
        self.estimators = estimators
        self.experimental_settings = experimental_settings
        self.keep_results = keep_results
        self.path = path
        self.folder = folder
        self.write_to_file = write_to_file
        self.channel_first = channel_first
        self.softmax = softmax
        self.device = device
        self.name = self.master.fname

        # Inits.
        self.results = {}

    def __call__(self, *args, **kwargs) -> dict:
        """Run (full) benchmarking exercise."""

        uid = str(uuid.uuid4())[:4]
        today = datetime.today().strftime("%d%m%Y")

        # Loop over datasets!
        for dataset_name, settings_data in self.experimental_settings.items():
            print(f"{dataset_name}")
            self.results[dataset_name] = {}

            # Loop over models!
            for (model_name, model), (_, gc_layer) in zip(
                settings_data["models"].items(), settings_data["gc_layers"].items()
            ):
                self.results[dataset_name][model_name] = {}
                print(f"  {model_name}")

                # Loop over estimators!
                for estimator_category, estimator_meta in self.estimators.items():
                    print(f"    {estimator_category}")
                    self.results[dataset_name][model_name][estimator_category] = {}

                    for estimator_name, (estimator) in estimator_meta.items():
                        print(f"      {estimator_name}")

                        self.results[dataset_name][model_name][estimator_category][
                            estimator_name
                        ] = {}

                        # Update attributes of master, make sure to save every run of master.
                        self.master.fname = f"{dataset_name}_{model_name}_{estimator_category}_{estimator_name}_{self.name}"
                        self.master.write_to_file = True
                        self.master.uid = uid
                        self.master.path = self.path + f"{dataset_name}/"

                        # Run full analysis.
                        self.master(
                            estimator=estimator[0],
                            model=model,
                            x_batch=settings_data["x_batch"],
                            y_batch=settings_data["y_batch"],
                            a_batch=None,
                            s_batch=settings_data["s_batch"],
                            channel_first=self.channel_first,
                            softmax=self.softmax,
                            device=self.device,
                            lower_is_better=estimator[1],
                        )

                        # Keep results.
                        if self.keep_results:
                            self.results[dataset_name][model_name][estimator_category][
                                estimator_name
                            ]["results_intra_scores"] = self.master.results_intra_scores
                            self.results[dataset_name][model_name][estimator_category][
                                estimator_name
                            ]["results_inter_scores"] = self.master.results_inter_scores
                            self.results[dataset_name][model_name][estimator_category][
                                estimator_name
                            ]["results_eval_scores"] = self.master.results_eval_scores
                            self.results[dataset_name][model_name][estimator_category][
                                estimator_name
                            ][
                                "results_eval_scores_perturbed"
                            ] = self.master.results_eval_scores_perturbed
                            self.results[dataset_name][model_name][estimator_category][
                                estimator_name
                            ][
                                "results_indices_perturbed"
                            ] = self.master.results_indices_perturbed
                            self.results[dataset_name][model_name][estimator_category][
                                estimator_name
                            ][
                                "results_meta_consistency_scores"
                            ] = self.master.results_meta_consistency_scores
                            self.results[dataset_name][model_name][estimator_category][
                                estimator_name
                            ][
                                "results_consistency_scores"
                            ] = self.master.results_consistency_scores

                        # Collect garbage.
                        gc.collect()
                        torch.cuda.empty_cache()

                        self.master.print_meta_consistency_scores()

                    # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                    # print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

            fname = f"{today}_{dataset_name}_benchmark_exercise_{uid}_{self.name}"

            # Remove dangling '_'.
            if fname.endswith("_"):
                fname = fname[:-1]

            dump_obj(
                obj=self.results,
                path=self.path + self.folder,
                fname=fname,
                use_json=True,
            )

        print(f"Benchmarking completed (stored in {self.path + self.folder + fname}).")

        return self.results
