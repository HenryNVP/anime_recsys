from __future__ import annotations
import numpy as np

def hr_ndcg_at_k(scores: np.ndarray, k: int):
    order = np.argsort(-scores, axis=1)
    topk = order[:, :k]
    hits = (topk == 0).any(axis=1).astype(float)

    pos = np.argwhere(order == 0)
    ranks = np.full(order.shape[0], order.shape[1], dtype=int)
    ranks[pos[:,0]] = pos[:,1]
    ndcg = np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0)
    return float(hits.mean()), float(ndcg.mean())
