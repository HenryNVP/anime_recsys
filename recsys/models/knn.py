import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

class UserKNN:
    def __init__(self, k: int = 50):
        self.k = k
        self.user_sim = None
        self.ui = None

    def fit(self, u_pos: np.ndarray, i_pos: np.ndarray, num_users: int, num_items: int):
        import scipy.sparse as sp
        self.ui = sp.csr_matrix((np.ones(len(u_pos)), (u_pos, i_pos)),
                                shape=(num_users, num_items))
        from sklearn.metrics.pairwise import cosine_similarity
        self.user_sim = cosine_similarity(self.ui)
        np.fill_diagonal(self.user_sim, 0.0)
        return self
    
    def score_user_candidates(self, u: int, candidates: np.ndarray) -> np.ndarray:
        if self.user_sim is None or self.ui is None:
            raise RuntimeError("Model has not been fit yet. Call .fit() before scoring.")
        sim = self.user_sim[u]
        topk_idx = np.argpartition(-sim, self.k)[:self.k]
        topk_sim = sim[topk_idx]
        neigh_inter = self.ui[topk_idx][:, candidates].toarray()
        scores = topk_sim @ neigh_inter
        return scores

