from typing import Dict
import numpy as np
import quantus
from quantus.metrics import *
from quantus.functions import (
    perturb_func,
    similarity_func,
    norm_func,
    normalise_func,
)


def setup_xai_methods(
    gc_layer: str,
    img_size: int = 28,
) -> Dict:
    return {
    "Gradient": {
    },
    "Saliency": {
    },
    "IntegratedGradients": {
    },
    "GradCAM": {
        "gc_layer": gc_layer,
        "interpolate": (img_size, img_size),
        "interpolate_mode": "bilinear",

    },
    }


def setup_estimators(
    features: int,
    num_classes: int,
    img_size: int,
    percentage: int,
) -> Dict:
    return {
        "Robustness": {
            "Max-Sensitivity": (quantus.MaxSensitivity(
                nr_samples=10,
                perturb_func=perturb_func.uniform_noise,
                norm_numerator=norm_func.fro_norm,
                norm_denominator=norm_func.fro_norm,
                lower_bound=0.1,
                abs=False,
                normalise=True,
                normalise_func=normalise_func.normalise_by_max,
                return_aggregate=False,
                aggregate_func=np.mean,
                disable_warnings=True,
            ), True),
            "Local Lipschitz Estimate": (quantus.LocalLipschitzEstimate(
                nr_samples=10,
                perturb_func=perturb_func.gaussian_noise,
                norm_numerator=similarity_func.distance_euclidean,
                norm_denominator=similarity_func.distance_euclidean,
                perturb_std=0.1,
                abs=False,
                normalise=True,
                normalise_func=normalise_func.normalise_by_max,
                return_aggregate=False,
                aggregate_func=np.mean,
                disable_warnings=True,
            ), True),
        },
        "Randomisation": {
            "Random Logit": (quantus.RandomLogit(
                similarity_func=similarity_func.correlation_spearman,
                num_classes=num_classes,
                abs=False,
                normalise=True,
                normalise_func=normalise_func.normalise_by_max,
                return_aggregate=False,
                aggregate_func=np.mean,
                disable_warnings=True,
            ), False),
            "Model Parameter Randomisation Test": (quantus.ModelParameterRandomisation(
                similarity_func=similarity_func.correlation_spearman,
                return_sample_correlation=True,
                abs=False,
                normalise=True,
                normalise_func=normalise_func.normalise_by_max,
                return_aggregate=False,
                aggregate_func=np.mean,
                disable_warnings=True,
            ), True),
        },
        "Faithfulness": {
            "Faithfulness Correlation": (quantus.FaithfulnessCorrelation(
                subset_size=features,
                perturb_baseline="uniform",
                perturb_func=perturb_func.baseline_replacement_by_indices,
                nr_runs=10,
                abs=False,
                normalise=True,
                normalise_func=normalise_func.normalise_by_max,
                return_aggregate=False,
                aggregate_func=np.mean,
                disable_warnings=True,
            ), False),
            "Pixel-Flipping": (quantus.PixelFlipping(
                features_in_step=features,
                perturb_baseline="uniform",
                perturb_func=perturb_func.baseline_replacement_by_indices,
                abs=False,
                normalise=True,
                normalise_func=normalise_func.normalise_by_max,
                return_aggregate=False,
                aggregate_func=np.mean,
                return_auc_per_sample=True,
                disable_warnings=True,
            ), False),
        },
        "Complexity": {
            "Sparseness": (quantus.Sparseness(
                abs=False,
                normalise=True,
                normalise_func=normalise_func.normalise_by_max,
                return_aggregate=False,
                aggregate_func=np.mean,
                disable_warnings=True,
            ), False),
            "Complexity": (quantus.Complexity(
                abs=False,
                normalise=True,
                normalise_func=normalise_func.normalise_by_max,
                return_aggregate=False,
                aggregate_func=np.mean,
                disable_warnings=True,
            ), True),
        },
        "Localisation": {
            "Pointing-Game": (quantus.PointingGame(
                abs=False,
                normalise=True,
                normalise_func=normalise_func.normalise_by_max,
                return_aggregate=False,
                aggregate_func=np.mean,
                disable_warnings=True,
            ), False),
            #"Top-K Intersection": (quantus.TopKIntersection(
            #    k=int((img_size*img_size)*percentage),
            #    abs=False,
            #    normalise=True,
            #    normalise_func=normalise_func.normalise_by_max,
            #    return_aggregate=False,
            #    aggregate_func=np.mean,
            #    disable_warnings=True,
            #), False),
            "Relevance Rank Accuracy": (quantus.RelevanceRankAccuracy(
                abs=False,
                normalise=True,
                normalise_func=normalise_func.normalise_by_max,
                return_aggregate=False,
                aggregate_func=np.mean,
                disable_warnings=True,
            ), False),
            #"Relevance Mass Accuracy": (quantus.RelevanceMassAccuracy(
            #    abs=False,
            #    normalise=True,
            #    normalise_func=normalise_func.normalise_by_max,
            #    return_aggregate=False,
            #    aggregate_func=np.mean,
            #    disable_warnings=True,
            #), False),
        },
    }
