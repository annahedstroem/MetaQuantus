from typing import Dict, Any, Tuple
import torch
import gc
import uuid
from datetime import datetime
from quantus import explain
from quantus.metrics.base import Metric, PerturbationMetric

from .base import Analyser
from .master import MasterAnalyser
from .utils import *


class BenchmarkEstimators:
    def __init__(
        self,
        master: MasterAnalyser,
        estimators: Dict[
            str, Dict[str, Tuple[Union[Metric, PerturbationMetric], bool]]
        ],
        experimental_settings: Dict[str, Dict[str, Any]],
        path: str = "/content/drive/MyDrive/Projects/MetaQuantus/results/",
        write_to_file: bool = True,
        keep_results: bool = False,
        channel_first: Optional[bool] = True,
        softmax: Optional[bool] = False,
        device: Optional[str] = None,
    ):
        self.master = master
        self.estimators = estimators
        self.experimental_settings = experimental_settings
        self.keep_results = keep_results
        self.path = path
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
                            ]["intra_scores"] = self.master.intra_scores
                            self.results[dataset_name][model_name][estimator_category][
                                estimator_name
                            ]["inter_scores"] = self.master.inter_scores
                            self.results[dataset_name][model_name][estimator_category][
                                estimator_name
                            ]["eval_scores"] = self.master.eval_scores
                            self.results[dataset_name][model_name][estimator_category][
                                estimator_name
                            ][
                                "eval_scores_perturbed"
                            ] = self.master.eval_scores_perturbed
                            self.results[dataset_name][model_name][estimator_category][
                                estimator_name
                            ]["indices_perturbed"] = self.master.indices_perturbed

                        # Collect garbage.
                        gc.collect()
                        torch.cuda.empty_cache()

                    # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                    # print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

            fname = f"{today}_{dataset_name}_benchmark_exercise_{uid}_{self.name}"

            # Remove dangling '_'.
            if fname.endswith("_"):
                fname = fname[:-1]

            dump_obj(
                obj=self.results,
                path=self.path + f"benchmarks/",
                fname=fname,
                use_json=True,
            )

        return self.results
