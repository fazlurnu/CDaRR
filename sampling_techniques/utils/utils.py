from typing import Any, Dict, List, Sequence, Tuple

def extract_features_and_labels(
    data: Sequence[Dict[str, Any]],
    ipr_threshold: float,
) -> Tuple[List[float], List[float], List[bool]]:
    """
    Returns:
        lookahead_times: x2_lookahead_time
        resofach_values: x1_resofach
        safe_labels: overall_ipr >= ipr_threshold
    """
    lookahead_times = [d["x2_lookahead_time"] for d in data]
    resofach_values = [d["x1_resofach"] for d in data]
    safe_labels = [d["sim_results"]["overall_ipr"] >= ipr_threshold for d in data]
    return lookahead_times, resofach_values, safe_labels