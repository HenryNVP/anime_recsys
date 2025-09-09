import numpy as np
import scipy.sparse as sp

class UserKNN:
    """
    Memory-based CF (user-user KNN) with on-the-fly sparse cosine.
    Stores only UI CSR + per-user L2 norms. No full similarity matrix.
    """
    def __init__(self, k_neighbors: int = 50):
        self.k = k_neighbors
        self.ui: sp.csr_matrix | None = None
        self.user_norms: np.ndarray | None = None

    def fit(self, u_pos: np.ndarray, i_pos: np.ndarray, num_users: int, num_items: int):
        self.ui = sp.csr_matrix((np.ones(len(u_pos), dtype=np.float32),
                                 (u_pos.astype(np.int32), i_pos.astype(np.int32))),
                                 shape=(num_users, num_items))
        # L2 norm per user (for cosine); add eps to avoid divide-by-zero
        self.user_norms = np.sqrt(np.asarray(self.ui.multiply(self.ui).sum(axis=1)).ravel()) + 1e-12
        return self

    def score_user_candidates(self, u: int, candidates: np.ndarray) -> np.ndarray:
        if self.ui is None or self.user_norms is None:
            raise RuntimeError("UserKNN not fitted. Call .fit() first.")
        # cosine similarity to all users: sim = (UI * u_row^T) / (||u|| * ||·||)
        urow = self.ui.getrow(u)                                      # 1 x I
        num = np.asarray((self.ui @ urow.T)).ravel()                               # U
        sim = num / (self.user_norms * (np.linalg.norm(urow.data) + 1e-12))
        sim[u] = 0.0                                                  # drop self

        # top-k neighbor indices
        k = min(self.k, sim.size)
        topk_idx = np.argpartition(-sim, k-1)[:k]
        topk_sim = sim[topk_idx]                                      # (k,)

        # neighbor interactions restricted to candidates
        neigh_inter = self.ui[topk_idx][:, candidates].toarray()      # (k, |C|)
        return topk_sim @ neigh_inter                                 # (|C|,)
