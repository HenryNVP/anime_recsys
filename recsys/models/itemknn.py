# recsys/models/itemknn.py
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp

class ItemKNN:
    """
    Item-Item KNN using cosine on implicit UI.
    Train: build UI (CSR), compute co-occ C = UI^T UI (sparse), normalize to cosine,
           keep top-N neighbors per item.
    Save:  top_idx (I, N), top_sim (I, N)
    Score: sum of sims between candidate j and user's seen items.
    """
    def __init__(self, max_neighbors: int = 200, eps: float = 1e-12):
        self.max_neighbors = max_neighbors
        self.eps = eps
        self.top_idx: np.ndarray | None = None  # (I, N), int32 (pad with -1)
        self.top_sim: np.ndarray | None = None  # (I, N), float32

    def fit(self, train_df: pd.DataFrame, num_users: int, num_items: int):
        u = train_df["u"].to_numpy(np.int32)
        i = train_df["i"].to_numpy(np.int32)
        UI = sp.csr_matrix((np.ones(len(u), dtype=np.float32), (u, i)), shape=(num_users, num_items))

        # Co-occurrence (items x items), sparse
        C = (UI.T @ UI).tocsr().astype(np.float32)
        C.setdiag(0)  # ignore self

        # cosine normalization using item degrees
        deg = np.asarray(UI.sum(axis=0)).ravel().astype(np.float32)  # users per item
        norm = np.sqrt(deg) + self.eps

        # Keep top-N neighbors per item
        N = min(self.max_neighbors, num_items - 1) if num_items > 1 else 0
        top_idx = -np.ones((num_items, N), dtype=np.int32)
        top_sim = np.zeros((num_items, N), dtype=np.float32)

        for j in range(num_items):
            col = C.getcol(j)
            nbr_idx = col.indices
            nbr_val = col.data / (norm[nbr_idx] * norm[j] + self.eps)  # cosine
            if nbr_idx.size == 0 or N == 0:
                continue
            if nbr_idx.size > N:
                keep = np.argpartition(-nbr_val, N-1)[:N]
            else:
                keep = np.arange(nbr_idx.size)
            sel_idx = nbr_idx[keep]
            sel_sim = nbr_val[keep]
            order = np.argsort(-sel_sim)
            top_idx[j, :order.size] = sel_idx[order]
            top_sim[j, :order.size] = sel_sim[order]

        self.top_idx, self.top_sim = top_idx, top_sim
        return self

    def save(self, out_dir: str):
        import os, numpy as np, json
        if self.top_idx is None or self.top_sim is None:
            raise RuntimeError("ItemKNN not fitted: top_idx or top_sim is None.")
        np.savez_compressed(
            os.path.join(out_dir, "itemknn_neighbors.npz"),
            top_idx=self.top_idx, top_sim=self.top_sim
        )
        with open(os.path.join(out_dir, "itemknn_meta.json"), "w") as f:
            json.dump({"max_neighbors": int(self.max_neighbors)}, f)

    @staticmethod
    def load(out_dir: str) -> "ItemKNN":
        import os, numpy as np, json
        pack = np.load(os.path.join(out_dir, "itemknn_neighbors.npz"))
        with open(os.path.join(out_dir, "itemknn_meta.json")) as f:
            meta = json.load(f)
        m = ItemKNN(max_neighbors=int(meta["max_neighbors"]))
        m.top_idx = pack["top_idx"]; m.top_sim = pack["top_sim"]
        return m

    def score_user_candidates(self, user_seen: np.ndarray, candidates: np.ndarray, use_k: int) -> np.ndarray:
        """
        Score candidate items for one user by summing similarity to user's seen items.
        We only use top-`use_k` neighbors precomputed (<= max_neighbors).
        """
        if self.top_idx is None or self.top_sim is None:
            raise RuntimeError("ItemKNN not fitted/loaded.")
        K = min(use_k, self.top_idx.shape[1])
        seen_set = set(int(x) for x in user_seen.tolist())
        out = np.zeros(len(candidates), dtype=np.float32)
        for t, cand in enumerate(candidates):
            nbrs = self.top_idx[cand, :K]
            sims = self.top_sim[cand, :K]
            mask = np.array([ (nbr != -1) and (nbr in seen_set) for nbr in nbrs ])
            if mask.any():
                out[t] = sims[mask].sum()
        return out
