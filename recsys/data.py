# recsys/data.py
from __future__ import annotations
import os, pickle
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- 1) ONE canonical splitter (no negatives) ----------
def prepare_splits(
    data_dir: str = "data",
    out_dir: str = "outputs",
    min_user_inter: int = 5,
    min_item_inter: int = 5,
    seed: int = 42,
):
    """
    Load -> filter -> reindex -> per-user 70/15/15 split.
    This is the SINGLE source of truth. No negatives here.
    Caches splits + mappings in `out_dir` for reuse.
    Returns (train_df, val_df, test_df, stats_dict)
    """
    os.makedirs(out_dir, exist_ok=True)

    cache = os.path.join(out_dir, "splits_cached.flag")
    if os.path.exists(cache):
        # Fast path: load cached splits/mappings
        train = pd.read_parquet(os.path.join(out_dir, "train.parquet"))
        val   = pd.read_parquet(os.path.join(out_dir, "val.parquet"))
        test  = pd.read_parquet(os.path.join(out_dir, "test.parquet"))
        with open(os.path.join(out_dir, "mappings.pkl"), "rb") as f:
            m = pickle.load(f)
        return train, val, test, {"num_users": m["num_users"], "num_items": m["num_items"]}

    # --- load raw ---
    anime   = pd.read_csv(os.path.join(data_dir, "anime.csv"))
    ratings = pd.read_csv(os.path.join(data_dir, "rating.csv"))
    # keep explicit interactions (drop -1 = not watched
    ratings = ratings[ratings["rating"] != -1].copy()

    # --- filter sparsity ---
    uc = ratings["user_id"].value_counts()
    ic = ratings["anime_id"].value_counts()
    ratings = ratings[
        ratings["user_id"].isin(uc[uc >= min_user_inter].index) &
        ratings["anime_id"].isin(ic[ic >= min_item_inter].index)
    ].copy()

    # --- reindex to 0..U-1 / 0..I-1 ---
    uid_map = {u:i for i,u in enumerate(ratings["user_id"].unique())}
    iid_map = {a:j for j,a in enumerate(ratings["anime_id"].unique())}
    ratings["u"] = ratings["user_id"].map(uid_map).astype("int64")
    ratings["i"] = ratings["anime_id"].map(iid_map).astype("int64")

    num_users = ratings["u"].nunique()
    num_items = ratings["i"].nunique()

    # --- per-user 70/15/15 split (fixed seed) ---
    rng = np.random.default_rng(seed)
    def split_user(g: pd.DataFrame):
        idx = np.arange(len(g)); rng.shuffle(idx)
        n = len(g)
        n_tr = max(1, int(0.70 * n))
        n_va = max(1, int(0.15 * n))
        tr = g.iloc[idx[:n_tr]]
        va = g.iloc[idx[n_tr:n_tr+n_va]]
        te = g.iloc[idx[n_tr+n_va:]]
        return tr, va, te

    parts = [split_user(g) for _, g in ratings.groupby("u", sort=False)]
    train = pd.concat([p[0] for p in parts], ignore_index=True)[["u","i"]]
    val   = pd.concat([p[1] for p in parts], ignore_index=True)[["u","i"]]
    test  = pd.concat([p[2] for p in parts], ignore_index=True)[["u","i"]]

    # --- cache to disk ---
    train.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    val.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    test.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
    with open(os.path.join(out_dir, "mappings.pkl"), "wb") as f:
        pickle.dump({"uids": uid_map, "iids": iid_map,
                     "num_users": num_users, "num_items": num_items}, f)
    open(cache, "w").close()

    return train, val, test, {"num_users": num_users, "num_items": num_items}

# ---------- 2) (Optional) build TRAIN pointwise pairs for neural/BCE ----------
def _user_pos_map(df: pd.DataFrame) -> Dict[int, set]:
    return df.groupby("u")["i"].apply(set).to_dict()

def build_pointwise_pairs(
    train: pd.DataFrame,
    num_items: int,
    neg_k: int = 4,
    seed: int = 42,
):
    """
    For NeuMF-like pointwise BCE training.
    Returns numpy arrays (users, items, labels) with 1 positive + neg_k negatives per positive.
    """
    rng = np.random.default_rng(seed)
    pos = _user_pos_map(train)
    all_items = np.arange(num_items, dtype=np.int64)

    users, items, labels = [], [], []
    train_local = train[["u","i"]].astype({"u":"int64","i":"int64"})
    for _, row in tqdm(train_local.iterrows(), total=len(train_local), desc="Build train pairs"):
        u = int(row["u"]); i = int(row["i"])
        users.append(u); items.append(i); labels.append(1)
        seen = pos.get(u, set())
        seen_arr = np.fromiter(seen, dtype=np.int64) if seen else np.empty(0, dtype=np.int64)
        pool = np.setdiff1d(all_items, seen_arr, assume_unique=True)
        k = min(neg_k, len(pool))
        if k > 0:
            negs = rng.choice(pool, size=k, replace=False)
        else:  # extremely rare
            negs = rng.choice(all_items, size=neg_k, replace=True)
        for j in negs:
            users.append(u); items.append(int(j)); labels.append(0)

    return (np.array(users, dtype=np.int64),
            np.array(items, dtype=np.int64),
            np.array(labels, dtype=np.float32))

# ---------- 3) (Optional) build eval candidates (true+negatives) ----------
def build_eval_candidates(
    train: pd.DataFrame,
    split: pd.DataFrame,
    num_items: int,
    num_negs: int = 99,
    seed: int = 123,
):
    """
    Candidate lists for ranking metrics:
    returns users, truths, candidates where candidates[row][0] is the TRUE item.
    """
    rng = np.random.default_rng(seed)
    pos = _user_pos_map(train)
    all_items = np.arange(num_items, dtype=np.int64)

    users = split["u"].to_numpy(dtype=np.int64)
    truths = split["i"].to_numpy(dtype=np.int64)
    cands: List[np.ndarray] = []
    for u, t in zip(users, truths):
        seen = pos.get(int(u), set())
        seen_arr = np.fromiter(seen, dtype=np.int64) if seen else np.empty(0, dtype=np.int64)
        pool = np.setdiff1d(all_items, seen_arr, assume_unique=True)
        k = min(num_negs, len(pool))
        negs = rng.choice(pool, size=k, replace=False) if k > 0 else rng.choice(all_items, size=num_negs, replace=True)
        cands.append(np.concatenate([[t], negs]))
    return users, truths, np.stack(cands, axis=0)
