from __future__ import annotations
import os, pickle
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

def prepare_splits(
    data_dir: str = "data",
    out_dir: str = "outputs",
    min_user_inter: int = 5,
    min_item_inter: int = 5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    os.makedirs(out_dir, exist_ok=True)

    cache_flag = os.path.join(out_dir, "splits_cached.flag")
    if os.path.exists(cache_flag):
        train = pd.read_parquet(os.path.join(out_dir, "train.parquet"))
        val   = pd.read_parquet(os.path.join(out_dir, "val.parquet"))
        test  = pd.read_parquet(os.path.join(out_dir, "test.parquet"))
        with open(os.path.join(out_dir, "mappings.pkl"), "rb") as f:
            m = pickle.load(f)
        return train, val, test, m

    ratings = pd.read_csv(os.path.join(data_dir, "rating.csv"))
    ratings = ratings[ratings["rating"] != -1].copy()

    # Filter sparsity
    uc = ratings["user_id"].value_counts()
    ic = ratings["anime_id"].value_counts()
    ratings = ratings[
        ratings["user_id"].isin(uc[uc >= min_user_inter].index) &
        ratings["anime_id"].isin(ic[ic >= min_item_inter].index)
    ].copy()

    # Reindex
    uid_map = {u:i for i,u in enumerate(ratings["user_id"].unique())}
    iid_map = {a:j for j,a in enumerate(ratings["anime_id"].unique())}
    ratings["u"] = ratings["user_id"].map(uid_map).astype("int64")
    ratings["i"] = ratings["anime_id"].map(iid_map).astype("int64")

    num_users = ratings["u"].nunique()
    num_items = ratings["i"].nunique()

    # Per-user 70/15/15
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in ratings.groupby("u", sort=False):
        idx = np.arange(len(g)); rng.shuffle(idx)
        n = len(g)
        n_tr = max(1, int(0.70*n))
        n_va = max(1, int(0.15*n))
        tr = g.iloc[idx[:n_tr]]
        va = g.iloc[idx[n_tr:n_tr+n_va]]
        te = g.iloc[idx[n_tr+n_va:]]
        parts.append((tr, va, te))
    train = pd.concat([p[0] for p in parts], ignore_index=True)[["u","i"]]
    val   = pd.concat([p[1] for p in parts], ignore_index=True)[["u","i"]]
    test  = pd.concat([p[2] for p in parts], ignore_index=True)[["u","i"]]

    # Save splits + mappings
    train.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    val.to_parquet(  os.path.join(out_dir, "val.parquet"),   index=False)
    test.to_parquet( os.path.join(out_dir, "test.parquet"),  index=False)
    with open(os.path.join(out_dir, "mappings.pkl"), "wb") as f:
        pickle.dump({"num_users": num_users, "num_items": num_items,
                     "uid_map": uid_map, "iid_map": iid_map}, f)
    open(cache_flag, "w").close()
    return train, val, test, {"num_users": num_users, "num_items": num_items}

def _user_pos_map(df: pd.DataFrame) -> Dict[int, set]:
    return df.groupby("u")["i"].apply(set).to_dict()

def build_eval_candidates(
    train: pd.DataFrame, split: pd.DataFrame, num_items: int,
    num_negs: int = 99, seed: int = 123,
):
    rng = np.random.default_rng(seed)
    pos = _user_pos_map(train)
    all_items = np.arange(num_items, dtype=np.int64)

    users = split["u"].to_numpy(np.int64)
    truths = split["i"].to_numpy(np.int64)
    cands: List[np.ndarray] = []
    for u, t in zip(users, truths):
        seen = pos.get(int(u), set())
        seen_arr = np.fromiter(seen, dtype=np.int64) if seen else np.empty(0, dtype=np.int64)
        pool = np.setdiff1d(all_items, seen_arr, assume_unique=True)
        k = min(num_negs, len(pool))
        negs = rng.choice(pool, size=k, replace=False) if k > 0 else rng.choice(all_items, size=num_negs, replace=True)
        cands.append(np.concatenate([[t], negs]))
    return users, truths, np.stack(cands, axis=0)
