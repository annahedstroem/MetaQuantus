"""This is a module that contains different functions to plot and format the result from different experiments."""

# This file is part of MetaQuantus.
# MetaQuantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# MetaQuantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with MetaQuantus. If not, see <https://www.gnu.org/licenses/>.

from typing import Dict, List
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy


def plot_multiple_estimator_area(
    benchmark: Dict,
    estimators: Dict,
    dataset_name: str,
    colours: Dict,
    save: bool,
    path: str,
    average_over: list = ["Model", "Input"],
) -> None:
    """
    Plot the outcome of the benchmarking exercise.

    Parameters
    ----------
    benchmark: dict
        The benchmarking data.
    estimators: dict
        The estimators used in the experiment.
    dataset_name: str
        The name of the dataset.
    colours: dict
        Dictionary of colours, based on the metrics.
    save: boolean
        Indicates if plots should be saved.
    path: str
        The path for saving the plot.
    average_over: list
        A list of spaces to average over.

    Returns
    -------
    None
    """
    fig, axs = plt.subplots(2, 5, sharex=True, figsize=(20, 8))

    for ex1, (estimator_category, metrics) in enumerate(estimators.items()):
        for ex2, estimator_name in enumerate(metrics):
            mc_scores = []
            for px, perturbation_type in enumerate(["Input", "Model"]):

                # Collect scores.
                scores = {
                    "IAC_NR": np.array(
                        benchmark[estimator_category][estimator_name][
                            "results_consistency_scores"
                        ][perturbation_type]["intra_scores_res"]
                    ).flatten(),
                    "IAC_AR": np.array(
                        benchmark[estimator_category][estimator_name][
                            "results_consistency_scores"
                        ][perturbation_type]["intra_scores_adv"]
                    ).flatten(),
                    "IEC_NR": np.array(
                        benchmark[estimator_category][estimator_name][
                            "results_consistency_scores"
                        ][perturbation_type]["inter_scores_res"]
                    ).flatten(),
                    "IEC_AR": np.array(
                        benchmark[estimator_category][estimator_name][
                            "results_consistency_scores"
                        ][perturbation_type]["inter_scores_adv"]
                    ).flatten(),
                }

                # Set values for m* and the actual values by the estimator.
                X_gt = [-1, 0, 1, 0]
                Y_gt = [0, 1, 0, -1]
                X_area = [-scores["IAC_AR"].mean(), 0, scores["IEC_AR"].mean(), 0]
                Y_area = [0, scores["IAC_NR"].mean(), 0, -scores["IEC_NR"].mean()]

                # Set the spaces to average the MC value over.
                if perturbation_type in average_over:
                    mc_score = np.mean(
                        [
                            scores["IAC_NR"].mean(),
                            scores["IEC_NR"].mean(),
                            scores["IAC_AR"].mean(),
                            scores["IEC_AR"].mean(),
                        ]
                    )
                    mc_scores.append(mc_score)

                if perturbation_type == "Input":
                    axs[ex2, ex1].fill(
                        X_area,
                        Y_area,
                        color=colours[estimator_name],
                        alpha=0.75,
                        label=perturbation_type,
                        edgecolor="black",
                    )
                else:
                    axs[ex2, ex1].fill(
                        X_area,
                        Y_area,
                        color=colours[estimator_name],
                        alpha=0.5,
                        label=perturbation_type,
                        hatch="/",
                        edgecolor="black",
                    )

                # Plot m*.
                if px == 1:
                    axs[ex2, ex1].fill(
                        X_gt, Y_gt, color="black", alpha=0.075, label="m*"
                    )

                # Annotate the labels.
                axs[ex2, ex1].annotate("${IAC}_{AR}$", (-1, 0), fontsize=12)
                axs[ex2, ex1].annotate("${IAC}_{NR}$", (-0.2, 0.8), fontsize=12)
                axs[ex2, ex1].annotate("${IEC}_{AR}$", (0.7, 0), fontsize=12)
                axs[ex2, ex1].annotate("${IEC}_{NR}$", (-0.2, -0.8), fontsize=12)

            # Labels.
            axs[ex2, ex1].set_xticklabels(
                ["", "1", "0.5", "0", "0.5", "1"], fontsize=14
            )
            axs[ex2, ex1].set_yticklabels(
                ["", "1", "", "0.5", "", "0", "", "0.5", "", "1", ""], fontsize=14
            )
            if estimator_name == "Model Parameter Randomisation Test":
                estimator_name = "Model Parameter Random."

            # Title and grids.
            axs[ex2, ex1].set_title(
                f"{estimator_name} ({np.array(mc_scores).flatten().mean():.4f})",
                fontsize=15,
            )
            axs[ex2, ex1].grid()
            axs[ex2, ex1].legend(loc="upper left")
            plt.grid()

    plt.tight_layout()
    if save:
        plt.savefig(path + "plots/" + f"full_area_graph_{dataset_name}.png", dpi=500)
    plt.show()


def plot_single_estimator_area(
    benchmark: dict,
    estimator_category: str,
    estimator_name: str,
    dataset_name: str,
    perturbation_type: str,
    colours: Dict,
    save: bool,
    path: str,
):
    """
    Plot the outcome of the benchmarking exercise.

    Parameters
    ----------
    benchmark: dict
        The benchmarking data.
    estimator_category: str
        The estimator category.
    estimator_name: str
        The estimator name.
    perturbation_type: str
        The perturbation type, either '' or ''.
    dataset_name: str
        The name of the dataset.
    colours: dict
        Dictionary of colours, based on the metrics.
    save: boolean
        Indicates if plots should be saved.
    path: str
        The path for saving the plot.

    Returns
    -------
    None
    """

    # Get scores.
    scores = {
        "IAC_NR": np.array(
            benchmark[estimator_category][estimator_name]["results_consistency_scores"][
                perturbation_type
            ]["intra_scores_res"]
        ).flatten(),
        "IAC_AR": np.array(
            benchmark[estimator_category][estimator_name]["results_consistency_scores"][
                perturbation_type
            ]["intra_scores_adv"]
        ).flatten(),
        "IEC_NR": np.array(
            benchmark[estimator_category][estimator_name]["results_consistency_scores"][
                perturbation_type
            ]["inter_scores_res"]
        ).flatten(),
        "IEC_AR": np.array(
            benchmark[estimator_category][estimator_name]["results_consistency_scores"][
                perturbation_type
            ]["inter_scores_adv"]
        ).flatten(),
    }
    mc_scores = np.array(
        [
            scores["IAC_NR"].mean(),
            scores["IEC_NR"].mean(),
            scores["IAC_AR"].mean(),
            scores["IEC_AR"].mean(),
        ]
    )

    # Se the basics.
    fig = plt.figure(figsize=(4, 4))
    ax = plt.axes()

    # Set values for m* and the actual values by the estimator.
    X_gt = [-1, 0, 1, 0]
    Y_gt = [0, 1, 0, -1]
    X_area = [-scores["IAC_AR"].mean(), 0, scores["IEC_AR"].mean(), 0]
    Y_area = [0, scores["IAC_NR"].mean(), 0, -scores["IEC_NR"].mean()]

    # Plot the fill.
    plt.fill(X_gt, Y_gt, color="black", alpha=0.1)
    plt.fill(
        X_area, Y_area, color=colours[estimator_name], alpha=0.8, edgecolor="black"
    )

    # Annotate.
    plt.annotate(
        "${IAC}_{AR}$" + f'={scores["IAC_AR"].mean():.2f}',
        (-scores["IAC_AR"].mean(), 0.1),
        fontsize=12,
    )
    plt.annotate(
        "${IAC}_{NR}$" + f'={scores["IAC_NR"].mean():.2f}',
        (-0.2, scores["IAC_NR"].mean() + 0.05),
        fontsize=12,
    )
    plt.annotate(
        "${IEC}_{AR}$" + f'={scores["IEC_AR"].mean():.2f}',
        (scores["IEC_AR"].mean(), 0.1),
        fontsize=12,
    )
    plt.annotate(
        "${IEC}_{NR}$" + f'={scores["IEC_NR"].mean():.2f}',
        (-0.2, -scores["IEC_NR"].mean() - 0.1),
        fontsize=12,
    )

    # Labels, titles and grids.
    plt.title(
        f"{estimator_name} ({np.mean(mc_scores):.3f}) — {perturbation_type}",
        fontsize=12,
    )
    plt.grid()
    ax.set_xticklabels(
        ["", "1", "", "0.5", "", "0", "", "0.5", "", "1", ""], fontsize=15
    )
    ax.set_yticklabels(
        ["", "1", "", "0.5", "", "0", "", "0.5", "", "1", ""], fontsize=15
    )

    # Limits.
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    plt.tight_layout()
    if save:
        plt.savefig(
            path + "plots/" + f"single_area_graph_{dataset_name}_{estimator_name}.png",
            dpi=500,
        )
    plt.show()


def plot_benchmarking_scatter_plots(
    dfs: Dict[str, pd.DataFrame], colours: Dict, save: bool, path: str
) -> None:
    """
    Plot the scatter plots for benchmarking.

    Parameters
    ----------
    dfs: dict
        A dictionary with benchmarking results (as pd.DataFrame).
    colours: dict
        Dictionary of colours, based on the metrics.
    save: boolean
        Indicates if plots should be saved.
    path: str
        The path for saving the plot.

    Returns
    -------
    None
    """
    legend_elements = [
        matplotlib.lines.Line2D(
            [],
            [],
            color="black",
            marker="^",
            linestyle="None",
            markersize=10,
            label="IPT",
        ),
        matplotlib.lines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markersize=10,
            label="MPT",
        ),
    ]
    markers = {"Input": "^", "Model": "o"}
    fig, ax = plt.subplots(1, len(dfs) * 2, figsize=(len(dfs) * 6, 4))

    round = 0
    for i, (dataset_name, df) in enumerate(dfs.items()):

        i = i + round
        # Plot scatter.
        for x, y, col, t in zip(
            df["IAC_{NR}"].values,
            df["IAC_{AR}"].values,
            df["Estimator"].values,
            df["Test"].values,
        ):
            ax[i].scatter(
                x=x,
                y=y,
                marker=markers[t],
                c=colours[col],
                s=100,
                label=t,
                alpha=0.65,
                edgecolor="black",
            )
        for x, y, col, t in zip(
            df["IEC_{NR}"].values,
            df["IEC_{AR}"].values,
            df["Estimator"].values,
            df["Test"].values,
        ):
            ax[i + 1].scatter(
                x=x,
                y=y,
                marker=markers[t],
                c=colours[col],
                s=100,
                label=t,
                alpha=0.65,
                edgecolor="black",
            )

        # Details for the first scatter.
        ax[i].legend(
            handles=legend_elements,
            fontsize=15,
            frameon=True,
            edgecolor="black",
            loc="lower left",
        )
        ax[i].set_ylabel("$\mathbf{IAC}_{AR}$", fontsize=20)
        ax[i].set_xlabel("$\mathbf{IAC}_{NR}$", fontsize=20)
        ax[i].set_ylim(-0.1, 1.1)
        ax[i].set_xlim(-0.1, 1.1)
        ax[i].set_xticks(
            ticks=np.linspace(0, 1, 10),
            labels=[0.0, "", "", "", "", "", "", "", "", 1.0],
            fontsize=15,
        )
        ax[i].set_yticks(
            ticks=np.linspace(0, 1, 10),
            labels=[0.0, "", "", "", "", 0.5, "", "", "", 1.0],
            fontsize=15,
        )
        ax[i].grid()
        ax[i].set_title(f"{dataset_name}", fontsize=16)

        # Then, the sceond scatter.
        ax[i + 1].legend(
            handles=legend_elements,
            fontsize=15,
            frameon=True,
            edgecolor="black",
            loc="lower left",
        )
        ax[i + 1].set_ylabel("$\mathbf{IEC}_{AR}$", fontsize=20)
        ax[i + 1].set_xlabel("$\mathbf{IEC}_{NR}$", fontsize=20)
        ax[i + 1].set_ylim(-0.1, 1.1)
        ax[i + 1].set_xlim(-0.1, 1.1)
        ax[i + 1].set_xticks(
            ticks=np.linspace(0, 1, 10),
            labels=[0.0, "", "", "", "", "", "", "", "", 1.0],
            fontsize=15,
        )
        ax[i + 1].set_yticks(
            ticks=np.linspace(0, 1, 10),
            labels=[0.0, "", "", "", "", 0.5, "", "", "", 1.0],
            fontsize=15,
        )
        ax[i + 1].grid()
        ax[i + 1].set_title(f"{dataset_name}", fontsize=16)

        round = i + 1

    plt.tight_layout()

    if save:
        datasets = (
            str(list(dfs.keys()))
            .replace("'", "")
            .replace("[", "")
            .replace("]", "")
            .replace(", ", "_")
        )
        plt.savefig(
            path + "plots/" + f"benchmarking_scatter_plot_{datasets}.png", dpi=1000
        )
    plt.show()


def plot_benchmarking_scatter_bar_plots_combined(
    df: pd.DataFrame, means: list, stds: list, colours: Dict, save: bool, path: str
) -> None:
    """
    Plot the scatter plots (left) and average MC scores (right).

    Parameters
    ----------
    df: pd.DataFrame
        The benchmarking results used for the scatter plots.
    means: list
        The means for the different datasets.
    stds: list
        The stds for the different datasets.
    colours: dict
        Dictionary of colours, based on the metrics.
    save: boolean
        Indicates if plots should be saved.
    path: str
        The path for saving the plot.

    Returns
    -------
    None
    """
    legend_elements = [
        matplotlib.lines.Line2D(
            [],
            [],
            color="black",
            marker="^",
            linestyle="None",
            markersize=10,
            label="IPT",
        ),
        matplotlib.lines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markersize=10,
            label="MPT",
        ),
    ]
    markers = {"Input": "^", "Model": "o"}  # ['s', '*', '^']

    fig, ax = plt.subplots(
        1, 3, figsize=(16, 4), gridspec_kw={"width_ratios": [1, 1, 4]}
    )

    # Plot the scatter plots.
    for x, y, col, t in zip(
        df["IAC_{NR}"].values,
        df["IAC_{AR}"].values,
        df["Estimator"].values,
        df["Test"].values,
    ):
        ax[0].scatter(
            x=x,
            y=y,
            marker=markers[t],
            c=colours[col],
            s=100,
            label=t,
            alpha=0.65,
            edgecolor="black",
        )
    for x, y, col, t in zip(
        df["IEC_{NR}"].values,
        df["IEC_{AR}"].values,
        df["Estimator"].values,
        df["Test"].values,
    ):
        ax[1].scatter(
            x=x,
            y=y,
            marker=markers[t],
            c=colours[col],
            s=100,
            label=t,
            alpha=0.65,
            edgecolor="black",
        )

    # Details for scatter 1.
    ax[0].legend(
        handles=legend_elements,
        fontsize=15,
        frameon=True,
        edgecolor="black",
        loc="lower left",
    )
    ax[0].set_ylabel("$\mathbf{IAC}_{AR}$", fontsize=20)
    ax[0].set_xlabel("$\mathbf{IAC}_{NR}$", fontsize=20)
    ax[0].set_ylim(-0.1, 1.1)
    ax[0].set_xlim(-0.1, 1.1)
    ax[0].set_xticks(
        ticks=np.linspace(0, 1, 10),
        labels=[0.0, "", "", "", "", "", "", "", "", 1.0],
        fontsize=15,
    )
    ax[0].set_yticks(
        ticks=np.linspace(0, 1, 10),
        labels=[0.0, "", "", "", "", 0.5, "", "", "", 1.0],
        fontsize=15,
    )
    ax[0].grid()

    # Details for scatter 2.
    ax[1].legend(
        handles=legend_elements,
        fontsize=15,
        frameon=True,
        edgecolor="black",
        loc="lower left",
    )
    ax[1].set_ylabel("$\mathbf{IEC}_{AR}$", fontsize=20)
    ax[1].set_xlabel("$\mathbf{IEC}_{NR}$", fontsize=20)
    ax[1].set_ylim(-0.1, 1.1)
    ax[1].set_xlim(-0.1, 1.1)
    ax[1].set_xticks(
        ticks=np.linspace(0, 1, 10),
        labels=[0.0, "", "", "", "", "", "", "", "", 1.0],
        fontsize=15,
    )
    ax[1].set_yticks(
        ticks=np.linspace(0, 1, 10),
        labels=[0.0, "", "", "", "", 0.5, "", "", "", 1.0],
        fontsize=15,
    )
    ax[1].grid()

    # Configs for barplot.
    datasets = ["MNIST", "fMNIST", "cMNIST"]
    nr_datasets = 3
    metrics_short = ["SP", "CO", "FC", "PF", "PG", "RMA", "RL", "MPR", "MS", "LLE"]
    colours_repeat = np.repeat(list(colours.values()), repeats=nr_datasets)
    legend_elements = [
        matplotlib.patches.Patch(facecolor="white", edgecolor="black", hatch="/"),
        matplotlib.patches.Patch(facecolor="white", edgecolor="black", hatch="*"),
        matplotlib.patches.Patch(facecolor="white", edgecolor="black"),
    ]
    x = [
        0,
        1,
        2,
        4,
        5,
        6,
        8,
        9,
        10,
        12,
        13,
        14,
        16,
        17,
        18,
        20,
        21,
        22,
        24,
        25,
        26,
        28,
        29,
        30,
        32,
        33,
        34,
        36,
        37,
        38,
    ]
    labels_ticks = list(range(1, np.max(x) + 1, nr_datasets + 1))
    labels_ticks[0] = 1.5

    # Plot!
    barlist = ax[2].bar(x, means, yerr=stds, alpha=0.85, edgecolor="black")

    # Alt 1.
    ax[2].bar_label(barlist, fmt="%.2f", label_type="edge", fontsize=10)

    # Fix the harches.
    for i in range(0, len(metrics_short) * nr_datasets, 3):
        barlist[i].set_hatch("/")
    for i in range(1, len(metrics_short) * nr_datasets, 3):
        barlist[i].set_hatch("*")
    for i in range(len(metrics_short) * nr_datasets):
        barlist[i].set_color(colours_repeat[i])
    for i in range(len(metrics_short) * nr_datasets):
        barlist[i].set_edgecolor("black")

    # Set the labels and titles.
    ax[2].set_xticks(ticks=labels_ticks, labels=metrics_short, fontsize=20)
    ax[2].set_ylabel("$\mathbf{MC}$", fontsize=20)
    ax[2].set_yticks(
        ticks=np.linspace(0.5, 1.0, 10),
        labels=[0.5, "", 0.6, "", 0.7, "", 0.8, "", 0.9, ""],
        fontsize=15,
    )
    ax[2].set_ylim(np.min(means) - 0.1, np.max(means) + 0.1)
    ax[2].legend(
        handles=legend_elements, labels=datasets, ncol=3, fontsize=15, loc="upper left"
    )
    ax[2].grid()

    plt.tight_layout()
    if save:
        plt.savefig(
            path + "plots/" + f"benchmarking_scatter_bar_plots_combined.png", dpi=500
        )
    plt.show()


from typing import Dict


def make_benchmarking_df_as_str(benchmark: Dict, estimators: Dict):
    """
    Create the benchmarking df.

    Parameters
    ----------
    benchmark: dict
        The benchmarking data.
    estimators: dict
        The estimators used in the experiment

    Returns
    -------
    df
    """
    df = pd.DataFrame(
        columns=[
            "Category",
            "Estimator",
            "Test",
            "MC_bar",
            "MC",
            "IAC_{NR}",
            "IAC_{AR}",
            "IEC_{NR}",
            "IEC_{AR}",
        ]
    )
    scores = ["IAC_{NR}", "IAC_{AR}", "IEC_{NR}", "IEC_{AR}"]

    row = 0
    for ex1, (estimator_category, metrics) in enumerate(estimators.items()):
        for ex2, estimator_name in enumerate(metrics):
            means_bar = []
            stds_bar = []
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
                    score = np.array(
                        benchmark[estimator_category][estimator_name][
                            "results_meta_consistency_scores"
                        ][perturbation_type]["consistency_scores"][s]
                    )
                    df.loc[row, s] = (
                        f"{score.mean():.3f}" + " $\pm$ " + f"{score.std() * 2:.3f}"
                    )
                    means.append(score.mean())
                    stds.append(score.std())

                mc_mean = benchmark[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_mean"]
                mc_std = benchmark[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_std"]
                df.loc[row, "MC"] = f"{mc_mean:.3f}" + " $\pm$ " + f"{mc_std * 2:.3f}"

                means.append(mc_mean)
                stds.append(mc_std)

                mc_bar_mean = benchmark[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_means"]
                mc_bar_std = benchmark[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_std"]

                means_bar.append(mc_bar_mean)
                stds_bar.append(mc_bar_std)

                if px == 0:
                    df.loc[row, "MC_bar"] = ""
                elif px == 1:
                    df.loc[row, "MC_bar"] = (
                        f"{np.mean(np.array(means_bar).flatten()):.3f}"
                        + " $\pm$ "
                        + f"{np.mean(np.array(stds_bar).flatten()) * 2:.3f}"
                    )

    return df


def make_benchmarking_df(benchmark: Dict, estimators: Dict, std_times: int = 2):
    """
    Create the benchmarking df.

    Parameters
    ----------
    benchmark: dict
        The benchmarking data.
    estimators: dict
        The estimators used in the experiment.
    std_times: integer
        The number of times to add the standard deviation.

    Returns
    -------
    df
    """
    df = pd.DataFrame(
        columns=[
            "Category",
            "Estimator",
            "Test",
            "IAC_{NR}",
            "IAC_{AR}",
            "IEC_{NR}",
            "IEC_{AR}",
            "MC",
            "IAC_{NR} std",
            "IAC_{AR} std",
            "IEC_{NR} std",
            "IEC_{AR} std",
            "MC std",
        ]
    )
    scores = ["IAC_{NR}", "IAC_{AR}", "IEC_{NR}", "IEC_{AR}"]

    row = 0
    for ex1, (estimator_category, metrics) in enumerate(estimators.items()):
        for ex2, estimator_name in enumerate(metrics):
            for px, perturbation_type in enumerate(["Model", "Input"]):

                row += ex1 + ex2 + px
                df.loc[row, "Test"] = perturbation_type
                df.loc[row, "Category"] = estimator_category
                df.loc[row, "Estimator"] = estimator_name
                for s in scores:
                    score = np.array(
                        benchmark[estimator_category][estimator_name][
                            "results_meta_consistency_scores"
                        ][perturbation_type]["consistency_scores"][s]
                    )
                    df.loc[row, s] = score.mean()
                    df.loc[row, s + " std"] = score.std() * std_times

                df.loc[row, "MC"] = benchmark[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_mean"]
                df.loc[row, "MC std"] = (
                    benchmark[estimator_category][estimator_name][
                        "results_meta_consistency_scores"
                    ][perturbation_type]["MC_std"]
                    * std_times
                )

    return df


def aggregate_benchmarking_datasets(
    benchmark_mnist: Dict,
    benchmark_fmnist: Dict,
    benchmark_cmnist: Dict,
    estimators: Dict,
    perturbation_types: List[str],
):
    """
    Aggregate benchmarking data over datasets.

    Parameters
    ----------
    benchmark_mnist: dict
        A dictionary of the benchmarking data.
    benchmark_fmnist: dict
        A dictionary of the benchmarking data.
    benchmark_cmnist: dict
        A dictionary of the benchmarking data.
    estimators: dict
        The estimators used in the experiment.
    perturbation_types: list
        A list of strings containing the 'Input' and/or 'Model'.

    Returns
    -------
    tuple
    """
    # Collect the data.
    mnist_means = []
    mnist_stds = []
    fmnist_means = []
    fmnist_stds = []
    cmnist_means = []
    cmnist_stds = []

    for ex1, (estimator_category, metrics) in enumerate(estimators.items()):
        for ex2, estimator_name in enumerate(metrics):
            mnist_means_per = []
            mnist_stds_per = []
            fmnist_means_per = []
            fmnist_stds_per = []
            cmnist_means_per = []
            cmnist_stds_per = []

            for px, perturbation_type in enumerate(perturbation_types):

                mnist_mean = benchmark_mnist[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_mean"]
                mnist_std = benchmark_mnist[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_std"]
                mnist_means_per.append(mnist_mean)
                mnist_stds_per.append(mnist_std)

                fmnist_mean = benchmark_fmnist[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_mean"]
                fmnist_std = benchmark_fmnist[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_std"]
                fmnist_means_per.append(fmnist_mean)
                fmnist_stds_per.append(fmnist_std)

                cmnist_mean = benchmark_cmnist[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_mean"]
                cmnist_std = benchmark_cmnist[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_std"]
                cmnist_means_per.append(cmnist_mean)
                cmnist_stds_per.append(cmnist_std)

            mnist_means.append(np.mean(mnist_means_per))
            mnist_stds.append(np.mean(mnist_stds_per))
            fmnist_means.append(np.mean(fmnist_means_per))
            fmnist_stds.append(np.mean(fmnist_stds_per))
            cmnist_means.append(np.mean(cmnist_means_per))
            cmnist_stds.append(np.mean(cmnist_stds_per))

    means = np.array(
        [[a, b, c] for a, b, c in zip(mnist_means, fmnist_means, cmnist_means)]
    ).flatten()
    stds = np.array(
        [[a, b, c] for a, b, c in zip(mnist_stds, fmnist_stds, cmnist_stds)]
    ).flatten()

    return means, stds


def compute_means_over_datasets(
    benchmark_mnist: dict,
    benchmark_fmnist: dict,
    benchmark_cmnist: dict,
    estimators: dict,
    perturbation_types: list,
    include_std: bool = True,
    std_times: int = 2,
):
    """
    Get the ranking data over datasets.

    Parameters
    ----------
    benchmark_mnist: dict
        A dictionary of the benchmarking data.
    benchmark_fmnist: dict
        A dictionary of the benchmarking data.
    benchmark_cmnist: dict
        A dictionary of the benchmarking data.
    estimators: dict
        The estimators used in the experiment.
    perturbation_types: list
        A list of strings containing the 'Input' and/or 'Model'.
    include_std: bool
        Indicates if we should take into account standard deviation in the ranking.
    std_times: integer
        The number of times to add the standard deviation.

    Returns
    -------
    tuple
    """
    # Collect the data.
    means_mnist = {}
    means_fmnist = {}
    means_cmnist = {}

    for ex1, (estimator_category, metrics) in enumerate(estimators.items()):

        means_mnist[estimator_category] = {}
        means_fmnist[estimator_category] = {}
        means_cmnist[estimator_category] = {}

        for px, perturbation_type in enumerate(perturbation_types):
            means_mnist[estimator_category][perturbation_type] = {}
            means_fmnist[estimator_category][perturbation_type] = {}
            means_cmnist[estimator_category][perturbation_type] = {}

            for ex2, estimator_name in enumerate(metrics):

                mnist_mean = benchmark_mnist[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_means"]
                mnist_std = benchmark_mnist[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_std"]

                if include_std:
                    values = np.array(
                        [
                            [m + std_times * mnist_std for m in mnist_mean],
                            [m - std_times * mnist_std for m in mnist_mean],
                        ]
                    ).flatten()
                else:
                    values = mnist_mean
                means_mnist[estimator_category][perturbation_type][
                    estimator_name
                ] = values

                fmnist_mean = benchmark_fmnist[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_means"]
                fmnist_std = benchmark_mnist[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_std"]

                if include_std:
                    values = np.array(
                        [
                            [m + std_times * fmnist_std for m in fmnist_mean],
                            [m - std_times * fmnist_std for m in fmnist_mean],
                        ]
                    ).flatten()
                else:
                    values = fmnist_mean
                means_fmnist[estimator_category][perturbation_type][
                    estimator_name
                ] = values

                cmnist_mean = benchmark_cmnist[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_means"]
                cmnist_std = benchmark_mnist[estimator_category][estimator_name][
                    "results_meta_consistency_scores"
                ][perturbation_type]["MC_std"]

                if include_std:
                    values = np.array(
                        [
                            [m + std_times * cmnist_std for m in cmnist_mean],
                            [m - std_times * cmnist_std for m in cmnist_mean],
                        ]
                    ).flatten()
                else:
                    values = cmnist_mean
                means_cmnist[estimator_category][perturbation_type][
                    estimator_name
                ] = values

    return means_mnist, means_fmnist, means_mnist


def compute_ranking_over_datasets(
    means_mnist: np.array,
    means_fmnist: np.array,
    means_cmnist: np.array,
    estimators: dict,
    perturbation_types: List,
):
    """
    Compute ranking over the datasets.

    Parameters
    ----------
    means_mnist: np.array
        The mean values for different estimators for MNIST dataset.
    means_fmnist: np.array
        The mean values for different estimators for fMNIST dataset.
    means_cmnist: np.array
        The mean values for different estimators for cMNIST dataset.
    estimators: dict
        The estimators used in the experiment.
    perturbation_types: list
        A list of strings containing the 'Input' and/or 'Model'.

    Returns
    -------

    """
    ranking = {}
    frac_1st_estimator_1st = []
    frac_2nd_estimator_1st = []

    for estimator_category in estimators:
        ranking[estimator_category] = {}

        for perturbation_type in perturbation_types:
            ranking[estimator_category][perturbation_type] = {}

            for means_ds, dataset_name in zip(
                [means_mnist, means_fmnist, means_cmnist], ["MNIST", "fMNIST", "cMNIST"]
            ):

                # Get the mean values for the first and second estimator in each category.
                ranking[estimator_category][perturbation_type][dataset_name] = {}
                estimator_names = list(
                    means_ds[estimator_category][perturbation_type].keys()
                )
                estimator_name_1, estimator_name_2 = (
                    estimator_names[0],
                    estimator_names[1],
                )
                values = np.array(
                    list(means_ds[estimator_category][perturbation_type].values())
                )
                results_estimator_1, results_estimator_2 = values[0], values[1]

                # Sort the data.
                rank_data = np.argsort(values.flatten(), axis=0)

                # Calculate how often estimator one is ranked first.
                rank_first = (rank_data.max() + 1) / 2
                estimator_1_first = 0
                for i in rank_data[: int(rank_first)]:
                    if i > rank_first:
                        estimator_1_first += 1

                # Append the results.
                ranking[estimator_category][perturbation_type][dataset_name][
                    "1st_estimator_first"
                ] = (estimator_1_first / rank_first)
                ranking[estimator_category][perturbation_type][dataset_name][
                    "2st_estimator_first"
                ] = 1 - (estimator_1_first / rank_first)

                # Save as simple lists.
                frac_1st_estimator_1st.append(estimator_1_first / rank_first)
                frac_2nd_estimator_1st.append(1 - (estimator_1_first / rank_first))

    return ranking, frac_1st_estimator_1st, frac_2nd_estimator_1st


def plot_top_ranking_distribution(
    frac_1st_estimator_1st: list,
    frac_2nd_estimator_1st: list,
    estimators: Dict,
    dataset_name: str,
    colours: Dict,
    save: bool,
    path: str,
) -> None:
    """

    Parameters
    ----------
    frac_1st_estimator_1st: list
        The fraction of the wins for the first estimators.
    frac_2nd_estimator_1st: list
        The fraction of the wins for the first estimators.
    estimators: dict
        The estimators used in the experiment.
    dataset_name: str
        The name of the dataset.
    colours: dict
        Dictionary of colours, based on the metrics.
    save: boolean
        Indicates if plots should be saved.
    path: str
        The path for saving the plot.
    save

    Returns
    -------

    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Init configs.
    datasets = ["MNIST", "fMNIST", "cMNIST"]
    nr_datasets = 3
    total_rankings = np.arange(len(estimators) * nr_datasets).astype(int)

    # Create barplot!
    barlist1 = plt.bar(
        total_rankings,
        frac_1st_estimator_1st,
        color=np.repeat(list(colours.values())[::2], repeats=nr_datasets * 1),
        alpha=0.75,
        label="Estimator1",
        edgecolor="black",
    )
    barlist2 = plt.bar(
        total_rankings,
        frac_2nd_estimator_1st,
        bottom=frac_1st_estimator_1st,
        color=np.repeat(list(colours.values())[1::2], repeats=nr_datasets),
        alpha=0.5,
        label="Estimator2",
        edgecolor="black",
    )

    # Create legend of the different datasets.
    ax.set_xticks([], [])
    ax.legend(ncol=4, fontsize=20)

    legend_elements = [
        matplotlib.patches.Patch(facecolor="white", edgecolor="black", hatch="/"),
        matplotlib.patches.Patch(facecolor="white", edgecolor="black", hatch="*"),
        matplotlib.patches.Patch(facecolor="white", edgecolor="black"),
    ]

    # Fix hateches.
    for i in range(0, len(total_rankings), 3):
        barlist1[i].set_hatch("/")
        barlist2[i].set_hatch("/")
    for i in range(1, len(total_rankings), 3):
        barlist1[i].set_hatch("*")
        barlist2[i].set_hatch("*")

    # Set lims and labels.
    plt.ylim(-0.05, 1.2)
    plt.yticks(
        ticks=np.linspace(0.0, 1.0, 20),
        labels=[
            "",
            "",
            "",
            "25%",
            "",
            "",
            "",
            "",
            "",
            "50%",
            "",
            "",
            "",
            "",
            "75%",
            "",
            "",
            "",
            "",
            "100%",
        ],
        fontsize=15,
    )
    plt.legend(
        handles=legend_elements, labels=datasets, ncol=3, fontsize=15, loc="upper left"
    )  # loc=2,
    plt.ylabel("% Top Rank", fontsize=16)
    plt.xlabel(
        "Complexity  Faithfulness  Localisation  Randomisation Robustness",
        fontsize=15,
    )

    plt.tight_layout()
    if save:
        plt.savefig(path + "plots/" + f"ranking_{dataset_name}.png", dpi=1000)
    plt.show()


def make_category_convergence_df(benchmark: Dict, estimators: Dict):
    """
    Create the category convergence df.

    Parameters
    ----------
    benchmark: dict
        The benchmarking data.
    estimators: dict
        The estimators used in the experiment.

    Returns
    -------
    df
    """
    # Create dictionary of scores.
    scores = {}

    for px, perturbation_type in enumerate(["Input", "Model"]):
        scores[perturbation_type] = {}
        for ex1, (estimator_category, metrics) in enumerate(estimators.items()):
            for ex2, estimator_name in enumerate(metrics):
                # Collect scores for each perturbation type.
                scores[perturbation_type][estimator_name] = {
                    "intra_scores_res": np.array(
                        benchmark[estimator_category][estimator_name][
                            "results_consistency_scores"
                        ][perturbation_type]["intra_scores_res"]
                    ).flatten(),
                    "intra_scores_adv": np.array(
                        benchmark[estimator_category][estimator_name][
                            "results_consistency_scores"
                        ][perturbation_type]["intra_scores_adv"]
                    ).flatten(),
                    "inter_scores_res": np.array(
                        benchmark[estimator_category][estimator_name][
                            "results_consistency_scores"
                        ][perturbation_type]["inter_scores_res"]
                    ).flatten(),
                    "inter_scores_adv": np.array(
                        benchmark[estimator_category][estimator_name][
                            "results_consistency_scores"
                        ][perturbation_type]["inter_scores_adv"]
                    ).flatten(),
                }

    df = pd.DataFrame(
        columns=[
            "Metric_1",
            "Metric_2",
            "Category_1",
            "Category_2",
            "Within-Category",
            "Type",
            "Failure Mode",
            "Criterion",
            "Spear. Corr",
        ]
    )

    row = 0
    for ex1, (estimator_category_1, metrics_1) in enumerate(estimators.items()):
        for ex2, (estimator_category_2, metrics_2) in enumerate(estimators.items()):
            for ex2, metric_1 in enumerate(metrics_1):
                for ex2, metric_2 in enumerate(metrics_2):
                    for kx, score_type in enumerate(scores["Model"][metric_1].keys()):
                        if metric_1 != metric_2:

                            # Update the row.
                            row += ex1 + ex2 + kx
                            df.loc[row, "Metric_1"] = metric_1
                            df.loc[row, "Metric_2"] = metric_2
                            df.loc[row, "Category_1"] = estimator_category_1
                            df.loc[row, "Category_2"] = estimator_category_2

                            # Create a boolean row to indicate within or outside of category.
                            if estimator_category_1 == estimator_category_2:
                                df.loc[row, "Within-Category"] = 1
                            if estimator_category_1 != estimator_category_2:
                                df.loc[row, "Within-Category"] = 0

                            # Compute the correlation coefficient.
                            df.loc[row, "Spear. Corr"] = scipy.stats.spearmanr(
                                scores["Model"][metric_1][score_type],
                                scores["Model"][metric_2][score_type],
                            )[1]

                            df.loc[row, "Type"] = (
                                score_type.replace("scores_", "")
                                .replace("intra", "IAC")
                                .replace("inter", "IEC")
                                .replace("res", "NR")
                                .replace("adv", "AR")
                            )
                            df.loc[row, "Failure Mode"] = (
                                score_type.replace("_scores_", "")
                                .replace("intra", "")
                                .replace("inter", "")
                                .replace("res", "NR")
                                .replace("adv", "AR")
                            )
                            df.loc[row, "Criterion"] = (
                                score_type.replace("_scores_", "")
                                .replace("intra", "IAC")
                                .replace("inter", "IEC")
                                .replace("res", "")
                                .replace("adv", "")
                            )

    return df


def plot_category_convergence(
    means: np.array, stds: np.array, save: bool, path: str
) -> None:
    """

    Parameters
    ----------
    means: np.array
        A numpy array with the means (correlation coefficients).
    stds: np.array
        A numpy array with the standard deviations.
    save: boolean
        Indicates if plots should be saved.
    path: str
        The path for saving the plot.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(1, 1, figsize=(7.75, 5))

    x = [0, 1, 2, 4, 5, 6]
    barlist = plt.bar(x, means, yerr=stds, alpha=0.85, edgecolor="black")  # /2,
    ax.bar_label(barlist, fmt="%.2f", label_type="edge", fontsize=18)

    for i in range(3):
        barlist[i].set_color("#3b719f")
    for i in range(3, 6):
        barlist[i].set_color("#c8aca9")
    for i in range(0, 6):
        barlist[i].set_edgecolor("black")

    barlist[0].set_hatch("/")
    barlist[3].set_hatch("/")
    barlist[1].set_hatch("*")
    barlist[4].set_hatch("*")

    labels = np.linspace(0.0, 1.0, 10)
    ax.set_ylim(np.min(means) - 0.1, np.max(means) + 0.6)

    legend_elements = [
        matplotlib.patches.Patch(facecolor="white", edgecolor="black", hatch="/"),
        matplotlib.patches.Patch(facecolor="white", edgecolor="black", hatch="*"),
        matplotlib.patches.Patch(facecolor="white", edgecolor="black"),
    ]

    ax.legend(
        handles=legend_elements,
        labels=["MNIST", "fMNIST", "cMNIST"],
        ncol=3,
        fontsize=15,
        loc="upper left",
    )
    ax.set_yticks(
        ticks=np.linspace(0, 1, 11),
        labels=[0.0, "", 0.2, "", 0.4, "", 0.6, "", 0.8, "", 1.0],
        fontsize=18,
    )
    ax.set_ylabel("Spearman Rank Correlation", fontsize=18)
    ax.set_xticks(
        ticks=x,
        labels=["", "Outside-Category", "", "", "Within-Category", ""],
        fontsize=18,
    )
    ax.grid()

    plt.tight_layout()
    if save:
        plt.savefig(path + "plots/" + f"category_convergence.png", dpi=1000)
    plt.show()


def plot_hp_bar(
    df_result: pd.DataFrame, dataset_name: str, save: bool, path: str
) -> None:
    """
    Plot the hp plot.

    Parameters
    ----------
    df_result: pd.DataFrame
        The saved meta-evaluation results for the hyperparameter optimisation exercise.
    dataset_name: string
        The name of the dataset.
    save: boolean
        Indicates if plots should be saved.
    path: str
        The path for saving the plot.

    Returns
    -------
    None
    """

    settings = np.arange(0, len(df_result), 1)
    mc_scores = df_result["MC Mean"].values
    mc_scores_std = df_result["MC Std"].values
    iac_nr = df_result["IAC_{NR} mean"].values
    iac_ar = df_result["IAC_{AR} mean"].values
    iec_nr = df_result["IEC_{NR} mean"].values
    iec_ar = df_result["IEC_{AR} mean"].values

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    barlist1 = ax.bar(
        settings, iec_nr, alpha=0.65, color="#5a86ad", label="IEC_NR", edgecolor="black"
    )  ##aaa662
    barlist2 = ax.bar(
        settings,
        iac_nr,
        alpha=0.65,
        bottom=iec_nr,
        color="gray",
        label="IAC_NR",
        hatch="",
        edgecolor="black",
    )  ##719f91
    barlist3 = ax.bar(
        settings,
        iec_ar,
        alpha=0.65,
        bottom=(iac_nr + iec_nr),
        color="#b1d1fc",
        label="IEC_AR",
        edgecolor="black",
    )  ##c87f89
    barlist4 = ax.bar(
        settings,
        iac_ar,
        alpha=0.65,
        bottom=(iec_ar + iac_nr + iec_nr),
        yerr=mc_scores_std,
        color="#c5c9c7",
        label="IAC_AR",
        hatch="",
        edgecolor="black",
    )  # 738595#afa88b

    # plt.plot(2+mc_scores, "-o", c="black")

    for i in settings:
        if mc_scores[i] == mc_scores.max():
            plt.annotate(
                xy=(i - 0.4, 1.7 + mc_scores[i]),
                text=f"{(str(mc_scores[i]))[:5]}",
                fontsize=15,
                c="#3b719f",
            )
        else:
            plt.annotate(
                xy=(i - 0.4, 1.7 + mc_scores[i]),
                text=f"{(str(mc_scores[i]))[:5]}",
                fontsize=15,
            )

        plt.vlines(x=i, ymin=0, ymax=1.6 + mc_scores[i], color="black")

    ax.set_yticks(
        ticks=np.linspace(0, 3, 15),
        labels=["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
        fontsize=18,
    )
    ax.set_xticks(
        ticks=settings, labels=[f"P{i + 1}" for i in range(len(df_result))], fontsize=20
    )
    ax.grid()

    plt.xlabel(" ", fontsize=18)
    ax.set_ylabel("Relative Scoring Criteria", fontsize=18)

    plt.ylim(0, 2.8)

    ax.legend(ncol=4, fontsize=15)
    plt.tight_layout()

    if save:
        plt.savefig(path + "plots/" + f"hp_{dataset_name}.png", dpi=1000)
    plt.show()
