import numpy as np
from sampling_techniques.models.tanh_feature_logit import TanhFeatureLogit

def pick_next_points_near_p05(model, next_data, k=4, balanced=True):
    """
    Pick the next k points closest to the decision boundary p=0.5.

    Requirements met:
      1) Inputs: model, next_data
      2) Outputs: (sample_points, remaining_next_data) where sample_points are removed

    Parameters
    ----------
    model : TanhFeatureLogit or dict
        - If dict: must be compatible with TanhFeatureLogit.from_dict(model)
        - If TanhFeatureLogit: used directly
    next_data : list[dict]
        Each dict must contain:
          - "x2_lookahead_time"
          - "x1_resofach"
    k : int
        Number of points to select (default 4)
    balanced : bool
        If True, tries to pick roughly half from each side of p=0.5 (when possible)

    Returns
    -------
    sample_points : list[dict]
    remaining_next_data : list[dict]
    """
    if not isinstance(next_data, list):
        raise TypeError("next_data must be a list of dicts")

    if len(next_data) == 0:
        return [], []

    k = min(int(k), len(next_data))
    if k <= 0:
        return [], list(next_data)

    # Accept either a ready model or a serialized dict
    if isinstance(model, dict):
        model = TanhFeatureLogit.from_dict(model)

    # Build feature matrix: [lookahead_time, resofach]
    X_next = np.column_stack([
        [d["x2_lookahead_time"] for d in next_data],
        [d["x1_resofach"]       for d in next_data],
    ]).astype(float)

    # Probability predictions
    p_next = np.asarray(model.predict_proba(X_next), dtype=float).reshape(-1)
    dist = np.abs(p_next - 0.5)

    order = np.argsort(dist)

    if not balanced:
        chosen_idx = order[:k]
    else:
        left = order[p_next[order] < 0.5]
        right = order[p_next[order] >= 0.5]

        k_left = k // 2
        k_right = k - k_left

        chosen = list(left[:k_left]) + list(right[:k_right])

        # Top up if one side didn't have enough
        if len(chosen) < k:
            for i in order:
                if i not in chosen:
                    chosen.append(int(i))
                if len(chosen) == k:
                    break

        chosen_idx = np.array(chosen, dtype=int)

    # Extract selected points
    sample_points = [next_data[i] for i in chosen_idx]

    # Remove selected points from next_data (without mutating original list)
    chosen_set = set(map(int, chosen_idx))
    remaining_next_data = [d for j, d in enumerate(next_data) if j not in chosen_set]

    return sample_points, remaining_next_data