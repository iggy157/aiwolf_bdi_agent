def normalize_score(raw_score: float, max_score: float = 10.0) -> float:
    return min(1.0, raw_score / max_score)
