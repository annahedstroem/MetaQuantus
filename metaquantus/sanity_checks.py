from typing import Union, Tuple, Optional
import numpy as np


def sanity_analysis(
    sanity_type: str,
    items: int,
    unperturbed_scores: Optional[list] = None,
    perturbed: bool = False,
):
    """
    Supporting function to get sanity scores depending on the sanity type.
        - If Estimator_Different: we sample scores from different distributions
        when perturbed vs unperturbed.
        - If Estimator_Same: we pass the unperturbed as the perturbed sample.

    Parameters
    ----------
    sanity_type: str
        A string that specifies what scores to produce.
    items: int
        The number of scores to produce.

    Returns
    -------
    A numpy array of scores.
    """

    if sanity_type == "Estimator_Different":
        if perturbed:
            return np.random.normal(
                loc=np.random.randint(0, 1),
                scale=1,
                size=items,
            ).tolist()
        return np.random.normal(
            loc=np.random.randint(-1000, -1),
            scale=1,
            size=items,
        ).tolist()

    elif sanity_type == "Estimator_Same":
        if perturbed:
            return unperturbed_scores
        return np.random.uniform(
            low=0,
            high=1,
            size=items,
        ).tolist()

    else:
        raise ValueError(
            "The sanity type ('sanity_check') can be either 'Estimator_Different' or 'Estimator_Same'."
        )


def sanity_analysis_under_perturbation(
    sanity_type: str,
    items: int,
    nr_perturbations: int,
    xai_methods: dict,
    unperturbed_scores=dict,
):
    """
    Iteratively produce scores for sanity analysis.

    Parameters
    ----------
    sanity_type: str
        A string that specifies what scores to produce.
    items: int
        The number of scores to produce.
    nr_perturbations: int
        The number of perturbations.
    xai_methods: dict
        A dictionary of the explanation methods.

    Returns
    -------

    """

    # Determine shape of results.
    scores_perturbed = {
        k: np.ndarray((nr_perturbations, items), dtype=float) for k in xai_methods
    }
    y_preds_perturbed = np.ones((nr_perturbations, items), dtype=bool)
    indices_perturbed = np.ones((nr_perturbations, items), dtype=bool)

    for p in range(nr_perturbations):
        for x, (method, explain_func_kwargs) in enumerate(xai_methods.items()):
            scores_perturbed[method][p] = sanity_analysis(
                sanity_type=sanity_type,
                items=items,
                perturbed=True,
                unperturbed_scores=unperturbed_scores[method],
            )

    return scores_perturbed, y_preds_perturbed, indices_perturbed
