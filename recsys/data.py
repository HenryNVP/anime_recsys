# recsys/data.py
from __future__ import annotations
import os, pickle
import numpy as np
import pandas as pd
from typing import Tuple, List
from tqdm import tqdm


# ---------------- Load splits & mappings ----------------

def load_mappings(clean_dir: str = "data_clean") -> dict:
    """Load user/item mappings (created by preprocess.py)."""
    with open(os.path.join(clean_dir, "mappings.pkl"), "rb") as f:
        return pickle.load(f)

def load_splits(clean_dir: str = "data_clean") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load train/val/test splits and mappings."""
    train = pd.read_parquet(os.path.join(clean_dir, "train.parquet"))
    val   = pd.read_parquet(os.path.join(clean_dir, "val.parquet"))
    test  = pd.read_parquet(os.path.join(clean_dir, "test.parquet"))
    maps  = load_mappings(clean_dir)
    stats = {"num_users": maps["num_users"], "num_items": maps["num_items"]}
    return train, val, test, stats


# ---------------- Candidate generation ----------------

def _user_pos_map(df: pd.DataFrame) -> dict[int, set[int]]:
    """Map user -> set of positive item ids."""
    return df.groupby("u")["i"].apply(set).to_dict()

def build_eval_candidates(
    train: pd.DataFrame,
    split: pd.DataFrame,
    num_items: int,
    num_negs: int = 99,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each (u, true_i) in split, create candidate set:
       [true_i, neg1, neg2, ..., negN]
    True item always at index 0.
    """
    rng = np.random.default_rng(seed)
    pos = _user_pos_map(train)
    all_items = np.arange(num_items)

    users = split["u"].to_numpy()
    truths = split["i"].to_numpy()
    cands: List[np.ndarray] = []

    for u, t in zip(users, truths):
        seen = pos.get(int(u), set())
        pool = np.setdiff1d(all_items, np.fromiter(seen, dtype=int), assume_unique=True)
        k = min(num_negs, len(pool))
        if k > 0:
            negs = rng.choice(pool, size=k, replace=False)
        else:
            negs = rng.choice(all_items, size=num_negs, replace=True)
        cand = np.concatenate([[t], negs])  # true first
        cands.append(cand)

    return users, truths, np.stack(cands)


# ---------------- Pointwise pairs for training ----------------

def build_pointwise_pairs(
    train: pd.DataFrame,
    num_items: int,
    neg_k: int = 4,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (u, i, y) triples for pointwise BCE training.
    Each positive -> sampled K negatives.
    """
    rng = np.random.default_rng(seed)
    pos = _user_pos_map(train)
    all_items = np.arange(num_items)

    users, items, labels = [], [], []
    for u, i in tqdm(train[["u", "i"]].itertuples(index=False, name=None),
                 total=len(train), desc="Build train pairs"):
        u = int(u); i = int(i)
        users.append(u); items.append(i); labels.append(1.0)

        seen = pos.get(u, set())
        pool = np.setdiff1d(all_items, np.fromiter(seen, dtype=int), assume_unique=True)
        k = min(neg_k, len(pool))
        if k > 0:
            negs = rng.choice(pool, size=k, replace=False)
        else:
            negs = rng.choice(all_items, size=neg_k, replace=True)
        for j in negs:
            users.append(u); items.append(int(j)); labels.append(0.0)

    return np.array(users), np.array(items), np.array(labels, dtype=np.float32)
