from typing import Tuple, Sequence

from sampling_techniques.models.tanh_feature_logit import TanhFeatureLogit
from sampling_techniques.models.logistic_regression import logistic_regression

def fit_model(
    lookahead_times: Sequence[float],
    resofach_values: Sequence[float],
    safe_labels: Sequence[bool],
) -> Tuple[float, TanhFeatureLogit]:
    """
    Fits logistic_regression + wraps params into TanhFeatureLogit.
    Returns (validation_accuracy, model)
    """
    acc, params = logistic_regression(
        list(lookahead_times),
        list(resofach_values),
        list(safe_labels),
    )
    return acc, TanhFeatureLogit.from_dict(params)
