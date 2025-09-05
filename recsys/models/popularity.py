# recsys/models/popularity.py
from __future__ import annotations
import numpy as np
import pandas as pd

class Popularity:
    def __init__(self):
        self.item_score: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame, num_items: int):
        counts = train_df["i"].value_counts()
        score = np.zeros(num_items, dtype=np.float32)
        score[counts.index.to_numpy()] = counts.to_numpy(dtype=np.float32)
        if score.max() > 0:
            score = score / score.max()
        self.item_score = score
        return self

    def save(self, out_dir: str):
        import os, numpy as np
        if self.item_score is None:
            raise RuntimeError("Popularity model needs to be fitted before saving.")
        np.save(os.path.join(out_dir, "pop_item_scores.npy"), self.item_score)

    @staticmethod
    def load(out_dir: str) -> "Popularity":
        import os, numpy as np
        m = Popularity()
        m.item_score = np.load(os.path.join(out_dir, "pop_item_scores.npy"))
        return m

    def score_candidates(self, candidates: np.ndarray) -> np.ndarray:
        if self.item_score is None:
            raise RuntimeError("Popularity not fitted.")
        return self.item_score[candidates]
