from typing import List, Optional, Dict
import pathlib
import numpy as np
import pandas as pd

# from .utils import load_obj


def get_resources_per_dataset(
    dataset_name: str,
    models: dict,
    estimators: Dict[str, List[str]] = {
        "Complexity": ["Sparseness", "Complexity"],
        "Faithfulness": ["Faithfulness Correlation", "Pixel-Flipping"],
        "Localisation": ["Pointing-Game", "Relevance Rank Accuracy"],
        "Randomisation": ["Random Logit", "Model Parameter Randomisation Test"],
        "Robustness": ["Max-Sensitivity", "Local Lipschitz Estimate"],
    },
    path_results: str = "/content/drive/MyDrive/Projects/analysers/dev/results/",
) -> dict:
    """Get resources per dataset."""

    # Get fpaths etc.
    fpaths = [
        str(i)
        for i in pathlib.Path(f"{path_results}{dataset_name}").glob("*")
        if i.is_file()
    ]
    model = models[dataset_name]

    resources = {}
    for category in estimators:
        for metric in estimators[category]:
            try:
                fname = [
                    f
                    for f in fpaths
                    if f.startswith(
                        f"{path_results}{dataset_name}/_results_{dataset_name}_{model}_{category}_{metric}_"
                    )
                ][0]
                resources[metric] = load_obj(path=fname, fname="", use_json=False)
            except:
                print(
                    f"ERROR: Couldn't find results file - {dataset_name} - metric {metric} ({category})."
                )
    return resources


def convert_summary_table_to_df(
    resource: dict,
    metrics: List[str],
    analysis_type: str = "intra",
    inter_metric: Optional[str] = None,
    analyser_suite: List[str] = [
        "Parameter Sensitivity Test",
        "Data Variability Test",
        "Model Adversary Test",
        "Explanation Adversary Test",
    ],
    desc: bool = False,
) -> pd.DataFrame:
    print(analysis_type)
    # if desc:
    # if analysis_type == "intra":
    #    print("... DV, PS = larger p-values are better")
    #    print("... MA, EA = smaller p-values are better\n")
    # elif analysis_type == "inter":
    #    print("... DV, PS, MA, EA = larger alphas are better = more consistent rankings of different XAI methods")

    if inter_metric:
        table = f"results_{inter_metric}_summary_table_"
    else:
        table = f"results_{analysis_type}_summary_table_"

    pds = []
    keys = []
    for metric in metrics:
        if metric in resource:
            try:
                data = resource[metric][table]
            except:
                print(
                    f"The resource {table} of metric {metric} does not exist. Check spelling."
                )
                data = None
        pds.append(pd.DataFrame(data))
        keys.append(metric)

    df = pd.concat(pds, keys=keys)

    return df[analyser_suite]


def get_results_from_parts(
    resources_parts: dict, metric: str, analysis_type: str = "intra"
):
    analysers = resources_parts[metric].keys()
    results = {}
    for analyser in analysers:

        if analysis_type == "inter":
            results_analyser = np.array(
                list(
                    resources_parts[metric][analyser][
                        f"results_{analysis_type}_analysis_"
                    ][analyser]
                )
            ).flatten()
        else:
            results_analyser = np.array(
                list(
                    resources_parts[metric][analyser][
                        f"results_{analysis_type}_analysis_"
                    ][analyser].values()
                )
            ).flatten()
        results[analyser] = {
            "mean": results_analyser.mean(),
            "std": results_analyser.std(),
        }
    return results


def append_inter_reliability_summary_tables(
    resource: dict,
    metrics: List[str],
    analyser_suite: List[str] = [
        "Parameter Sensitivity Test",
        "Data Variability Test",
        "Model Adversary Test",
        "Explanation Adversary Test",
    ],
) -> None:
    inter_metrics = ["spearmans", "average_cohen_kappa"]

    for metric in metrics:
        for inter_metric in inter_metrics:
            resource[metric][f"results_{inter_metric}_summary_table_"] = {}
            for analyser in analyser_suite:
                results = resource[metric][f"results_{inter_metric}_"][analyser]
                resource[metric][f"results_{inter_metric}_summary_table_"][analyser] = {
                    "mean": results.mean(),
                    "std": results.std(),
                }
    return resource
