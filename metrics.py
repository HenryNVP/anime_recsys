import numpy as np
from typing import Tuple

def hr_ndcg_at_k(scores: np.ndarray, k: int = 10) -> Tuple[float, float]:
    """
    scores: array shape (N, C). For each row, candidate at index 0 is the TRUE item.
    Returns (HR@k, NDCG@k).
    """
    order = np.argsort(-scores, axis=1)          # rank desc
    topk = order[:, :k]
    hits = (topk == 0).any(axis=1).astype(np.float32)

    # position of the true item (index 0) in each row's ranking
    pos = np.argwhere(order == 0)                # [row, rank]
    ranks = np.full(order.shape[0], order.shape[1], dtype=np.int64)
    ranks[pos[:, 0]] = pos[:, 1]
    ndcg = np.where(hits > 0, 1.0 / np.log2(ranks + 2), 0.0)

    return float(hits.mean()), float(ndcg.mean())