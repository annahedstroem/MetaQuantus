import quantus
import typing
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Estimator:
    name: str
    category: str
    score_direction: bool
    init: quantus.Metric


def create_estimator_dict(
    estimators: typing.List[Estimator],
) -> typing.Dict[str, typing.Dict[str, typing.Dict[str, typing.Any]]]:
    """
    Create a dictionary of estimators.

    Parameters
    ----------
    estimators : list of Estimator
        The list of Estimator objects.

    Returns
    -------
    dict of str to dict of str to dict of str to any
        The dictionary of estimators with the following format:
        {category: {"name_of_estimator": {"init": ..., "score_direction": ...}}}

    Examples
    --------
    >>> estimator1 = Estimator(name="BRIDGE",
    ...                         category="Unified",
    ...                         score_direction="higher",
    ...                         init=quanuts.Bridge(nr_models=samples,
    ...                         perturbation_levels=perturbation_levels,
    ...                         similarity_func=similarity_func,
    ...                         dist_preds=measure_func,
    ...                         dist_expls=measure_func,
    ...                         nr_classes=num_classes,
    ...                         nr_levels=nr_levels,
    ...                         abs=abs,
    ...                         normalise=normalise,
    ...                         normalise_func=normalise_func,
    ...                         return_aggregate=return_aggregate,
    ...                         aggregate_func=np.mean,
    ...                         disable_warnings=disable_warnings))
    >>> estimator2 = Estimator(name="Model Parameter Randomisation Test",
    ...                         category="Randomisation",
    ...                         score_direction="lower",
    ...                         init=quanuts.ModelParameterRandomisation(nr_models=samples,
    ...                         perturbation_levels=perturbation_levels,
    ...                         similarity_func=similarity_func,
    ...                         dist_preds=measure_func,
    ...                         dist_expls=measure_func,
    ...                         nr_classes=num_classes,
    ...                         nr_levels=nr_levels,
    ...                         abs=abs,
    ...                         normalise=normalise,
    ...                         normalise_func=normalise_func,
    ...                         return_aggregate=return_aggregate,
    ...                         aggregate_func=np.mean,
    ...                         disable_warnings=disable_warnings))
    >>> estimators = [estimator1, estimator2]
    >>> create_estimator_dict(estimators)
    {'Unified': {'BRIDGE': {'init': <quanuts.Bridge object at 0x7f7d54cf60d0>,
                            'score_direction': "higher"}},
     'Randomisation': {'Model Parameter Randomisation Test': {'init': <quanuts.ModelParameterRandomisation object at 0x7f7d54cf6af0>,
                                                               'score_direction': "lower"}}}
    """
    estimator_dict = {}
    for estimator in estimators:
        if estimator.category not in estimator_dict:
            estimator_dict[estimator.category] = {}
        estimator_dict[estimator.category][estimator.name] = {
            "init": estimator.init,
            "score_direction": estimator.score_direction,
        }
    return estimator_dict
