from typing import Dict
import numpy as np
import pandas as pd


def make_category_convergence_df(benchmark: Dict,
                   estimators: Dict):
    # Create dictionary of scores.
    scores = {}

    for px, perturbation_type in enumerate(["Input", "Model"]):
        scores[perturbation_type] = {}
        for ex1, (estimator_category, metrics) in enumerate(estimators.items()):
            for ex2, estimator_name in enumerate(metrics):
                scores[perturbation_type][estimator_name] = {"intra_scores_res": np.array(
                    benchmark[estimator_category][estimator_name]["results_consistency_scores"][perturbation_type][
                        "intra_scores_res"]).flatten(),
                                                             "intra_scores_adv": np.array(
                                                                 benchmark[estimator_category][estimator_name][
                                                                     "results_consistency_scores"][perturbation_type][
                                                                     "intra_scores_adv"]).flatten(),
                                                             "inter_scores_res": np.array(
                                                                 benchmark[estimator_category][estimator_name][
                                                                     "results_consistency_scores"][perturbation_type][
                                                                     "inter_scores_res"]).flatten(),
                                                             "inter_scores_adv": np.array(
                                                                 benchmark[estimator_category][estimator_name][
                                                                     "results_consistency_scores"][perturbation_type][
                                                                     "inter_scores_adv"]).flatten(), }

    df = pd.DataFrame(columns=["Metric_1", "Metric_2", "Category_1", "Category_2", "Within-Category", "Type", "Failure Mode", "Criterion", "Norm", "Spear. Corr", "Pear. Corr"])

    row = 0
    for ex1, (estimator_category_1, metrics_1) in enumerate(estimators.items()):
        for ex2, (estimator_category_2, metrics_2) in enumerate(estimators.items()):
            for ex2, metric_1 in enumerate(metrics_1):
                for ex2, metric_2 in enumerate(metrics_2):
                    for kx, score_type in enumerate(scores["Model"][metric_1].keys()):
                        if metric_1 != metric_2:
                            #print(f'{metric_1} vs {metric_2} \tNorm: {np.linalg.norm(scores["Model"][metric_1][k]-scores["Model"][metric_2][k])}')
                            row += ex1+ex2+kx
                            df.loc[row, "Metric_1"] = metric_1
                            df.loc[row, "Metric_2"] = metric_2
                            df.loc[row, "Category_1"] = estimator_category_1
                            df.loc[row, "Category_2"] = estimator_category_2
                            if estimator_category_1 == estimator_category_2:
                                df.loc[row, "Within-Category"] = 1
                            if estimator_category_1 != estimator_category_2:
                                df.loc[row, "Within-Category"] = 0
                            df.loc[row, "Norm"] = np.linalg.norm(scores["Model"][metric_1][score_type]-scores["Model"][metric_2][score_type])
                            df.loc[row, "Spear. Corr"] = scipy.stats.spearmanr(scores["Model"][metric_1][score_type], scores["Model"][metric_2][score_type])[1]
                            df.loc[row, "Pear. Corr"] = scipy.stats.pearsonr(scores["Model"][metric_1][score_type], scores["Model"][metric_2][score_type])[1]
                            df.loc[row, "Type"] = score_type.replace("scores_", "").replace("intra", "IAC").replace("inter", "IEC").replace("res", "NR").replace("adv", "AR")
                            df.loc[row, "Failure Mode"] = score_type.replace("_scores_", "").replace("intra", "").replace("inter", "").replace("res", "NR").replace("adv", "AR")
                            df.loc[row, "Criterion"] = score_type.replace("_scores_", "").replace("intra", "IAC").replace("inter", "IEC").replace("res", "").replace("adv", "")

    return df

def make_summary_df(benchmark: dict, estimators: dict):
    df = pd.DataFrame(
        columns=["Category", "Estimator", "Test", "IAC_{NR}", "IAC_{AR}", "IEC_{NR}", "IEC_{AR}", "MC", "IAC_{NR} std",
                 "IAC_{AR} std", "IEC_{NR} std", "IEC_{AR} std", "MC std"])
    scores = ["IAC_{NR}", "IAC_{AR}", "IEC_{NR}", "IEC_{AR}"]

    row = 0
    for ex1, (estimator_category, metrics) in enumerate(estimators.items()):
        for ex2, estimator_name in enumerate(metrics):
            for px, perturbation_type in enumerate(["Model", "Input"]):

                if estimator_category in benchmark:
                    if estimator_name in benchmark[estimator_category]:
                        row += ex1 + ex2 + px
                        df.loc[row, "Test"] = perturbation_type
                        df.loc[row, "Category"] = estimator_category
                        df.loc[row, "Estimator"] = estimator_name

                        for s in scores:
                            score = np.array(
                                benchmark[estimator_category][estimator_name]["results_meta_consistency_scores"][
                                    perturbation_type]["consistency_scores"][s])
                            df.loc[row, s] = score.mean()
                            df.loc[row, s + " std"] = score.std() * 2

                        df.loc[row, "MC"] = \
                        benchmark[estimator_category][estimator_name]["results_meta_consistency_scores"][
                            perturbation_type]["MC_mean"]
                        df.loc[row, "MC std"] = \
                        benchmark[estimator_category][estimator_name]["results_meta_consistency_scores"][
                            perturbation_type]["MC_std"] * 2

    return df

def make_benchmarking_df(benchmark: Dict,
                         estimators: Dict):
    df = pd.DataFrame(columns=["Category", "Estimator", "Test", "IAC_{NR}", "IAC_{AR}", "IEC_{NR}", "IEC_{AR}", "MC"])
    scores = ["IAC_{NR}", "IAC_{AR}", "IEC_{NR}", "IEC_{AR}"]

    means_all = []
    stds_all = []
    row = 0
    for ex1, (estimator_category, metrics) in enumerate(estimators.items()):
        for ex2, estimator_name in enumerate(metrics):
            for px, perturbation_type in enumerate(["Model", "Input"]):

                means = []
                stds = []

                row += ex1 + ex2 + px
                df.loc[row, "Test"] = perturbation_type
                if px == 1:
                    df.loc[row, "Category"] = estimator_category
                    df.loc[row, "Estimator"] = estimator_name
                else:
                    df.loc[row, "Category"] = estimator_category
                    df.loc[row, "Estimator"] = estimator_name

                for s in scores:
                    score = np.array(benchmark[estimator_category][estimator_name]["results_meta_consistency_scores"][
                                         perturbation_type]["consistency_scores"][s])
                    df.loc[row, s] = f"{score.mean():.3f}" + " $\pm$ " + f"{score.std() * 2:.3f}"
                    means.append(score.mean())
                    stds.append(score.std())

                mc_mean = \
                benchmark[estimator_category][estimator_name]["results_meta_consistency_scores"][perturbation_type][
                    "MC_mean"]
                mc_std = \
                benchmark[estimator_category][estimator_name]["results_meta_consistency_scores"][perturbation_type][
                    "MC_std"]
                df.loc[row, "MC"] = f"{mc_mean:.3f}" + " $\pm$ " + f"{mc_std * 2:.3f}"

                means.append(mc_mean)
                stds.append(mc_std)

                means_all.append(means)
                stds_all.append(stds)

    means_all = np.array(means_all)
    stds_all = np.array(stds_all)

    return df


def cleanup_benchmarking_df(df: pd.DataFrame,
                            estimators: Dict):
    row = 0
    for ex1, (estimator_category, metrics) in enumerate(estimators.items()):
        for ex2, estimator_name in enumerate(metrics):
            for px, perturbation_type in enumerate(["Model", "Input"]):

                row += ex1 + ex2 + px
                if px == 1:
                    df.loc[row, "Category"] = ""
                    df.loc[row, "Estimator"] = ""
                else:
                    df.loc[row, "Category"] = "\multirow{ 4}{*}{\textit{" + estimator_category + "}}"
                    df.loc[row, "Estimator"] = "\multirow{ 2}{*}{\textit{" + estimator_name + "}}"

    df.drop(columns="Test", inplace=True)
    return df