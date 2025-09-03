import os, pickle
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm


# Load data
def load_anime_ratings(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    anime = pd.read_csv(os.path.join(data_dir, "anime.csv"))
    ratings = pd.read_csv(os.path.join(data_dir, "rating.csv"))
    ratings = ratings[ratings["rating"] != -1].copy()  # drop "not rated"
    return anime, ratings

#  Drop users and itemss with too few interactions
def filter_sparse(ratings: pd.DataFrame, min_user_inter: int = 5, min_item_inter: int = 5,) -> pd.DataFrame:
    uc = ratings["user_id"].value_counts()
    ic = ratings["anime_id"].value_counts()

    keep_users = uc[uc >= min_user_inter].index
    keep_items = ic[ic >= min_item_inter].index

    filtered = ratings[
        ratings["user_id"].isin(keep_users) &
        ratings["anime_id"].isin(keep_items)
    ].copy()
    return filtered

# Reindexing to contiguous ids
def reindex_ids(ratings: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int,int], Dict[int,int]]:
    uid_map = {uid: idx for idx, uid in enumerate(ratings["user_id"].unique())}
    iid_map = {aid: idx for idx, aid in enumerate(ratings["anime_id"].unique())}
    ratings_idx = ratings.copy()
    ratings_idx["u"] = ratings_idx["user_id"].map(uid_map)
    ratings_idx["i"] = ratings_idx["anime_id"].map(iid_map)
    return ratings_idx, uid_map, iid_map

# stratified split per user: 70% train, 15% val, 15% test
# ensure each user has at least one interaction in each split
def split_per_user(ratings_idx: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)
    parts = []
    for u, g in ratings_idx.groupby("u"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = max(1, int(0.70*n))
        n_val   = max(1, int(0.15*n))
        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train+n_val]
        test_idx  = idx[n_train+n_val:]
        # ensure non-empty splits
        if len(test_idx) == 0 and len(val_idx) > 1:
            test_idx = np.array([val_idx[-1]])
            val_idx = val_idx[:-1]
        parts.append((
            ratings_idx.loc[train_idx],
            ratings_idx.loc[val_idx],
            ratings_idx.loc[test_idx]
        ))
    train = pd.concat([p[0] for p in parts], ignore_index=True)
    val   = pd.concat([p[1] for p in parts], ignore_index=True)
    test  = pd.concat([p[2] for p in parts], ignore_index=True)
    return train, val, test


# Items per user
def _user_pos_map(df: pd.DataFrame):
    return df.groupby("u")["i"].apply(set).to_dict()

# Build user[], items[], ratings[] pairs
# rating = 1 for positive, 0 for negative
# For each positive interaction, sample neg_k negatives from unseen items
def build_pointwise_pairs(train, num_items, neg_k=4, seed=42):
    import numpy as np
    from tqdm import tqdm

    rng = np.random.default_rng(seed)
    pos = _user_pos_map(train)
    all_items = np.arange(num_items)

    users, items, labels = [], [], []
    train_local = train[["u", "i"]].astype({"u": "int64", "i": "int64"})

    for _, row in tqdm(train_local.iterrows(), total=len(train_local), desc="Build train pairs"):
        u = int(row["u"])
        i = int(row["i"])

        users.append(u); items.append(i); labels.append(1)

        seen = pos.get(u, set())
        pool = np.setdiff1d(all_items, np.fromiter(seen, dtype=int), assume_unique=True)
        k = min(neg_k, len(pool))
        negs = rng.choice(pool, size=k, replace=False) if k > 0 else rng.choice(all_items, size=neg_k, replace=True)

        for j in negs:
            users.append(u); items.append(int(j)); labels.append(0)

    return np.array(users), np.array(items), np.array(labels, dtype=np.float32)

#For each (user, true_item) in val/test, create a small candidate set: [ true_item, neg1, neg2, ... ]
def build_eval_candidates(
    train: pd.DataFrame,
    split: pd.DataFrame,
    num_items: int,
    num_negs: int = 99,
    seed: int = 123,
):
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

# 5) ONE-CALL DRIVER: PREPARE + SAVE EVERYTHING
# =============================================================================

def prepare_all(
    data_dir="data",
    out_dir="outputs",
    min_user_inter=5,
    min_item_inter=5,
    neg_k=4,
    num_eval_negs=99,
    seed=42,
):
    """
    One function to call from your training script.
    It will:
      1) Load + clean
      2) Filter sparse users/items
      3) Reindex IDs to 0..N-1
      4) Per-user 70/15/15 split
      5) Build training pairs (with negative sampling)
      6) Build val/test candidate sets (true + 99 negatives)
      7) Save everything under `out_dir`

    Files written:
      - train_pairs.npz         (u, i, y) for BCE training
      - val_users.npy,  val_cands.npy,  val_truths.npy
      - test_users.npy, test_cands.npy, test_truths.npy
      - mappings.pkl            (uid_map, iid_map, num_users, num_items)
      - items_tfidf.npz, tfidf.pkl   (for content baseline)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load
    anime, ratings = load_anime_ratings(data_dir)

    # 2) Filter sparse -> better embeddings
    ratings = filter_sparse(ratings, min_user_inter, min_item_inter)

    # 3) Reindex to contiguous ids
    ratings_idx, uid_map, iid_map = reindex_ids(ratings)
    num_users = ratings_idx["u"].nunique()
    num_items = ratings_idx["i"].nunique()

    # 4) Per-user split
    train, val, test = split_per_user(ratings_idx, seed=seed)

    # 5) Training (pointwise) pairs with negatives
    tr_u, tr_i, tr_y = build_pointwise_pairs(train, num_items, neg_k=neg_k, seed=seed)

    # 6) Candidate sets for ranking eval (val/test)
    vu, vt, vc = build_eval_candidates(train, val,  num_items, num_negs=num_eval_negs, seed=seed+1)
    tu, tt, tc = build_eval_candidates(train, test, num_items, num_negs=num_eval_negs, seed=seed+2)

    # Save arrays
    np.savez_compressed(os.path.join(out_dir, "train_pairs.npz"), u=tr_u, i=tr_i, y=tr_y)
    np.save(os.path.join(out_dir, "val_users.npy"),  vu)
    np.save(os.path.join(out_dir, "val_truths.npy"), vt)
    np.save(os.path.join(out_dir, "val_cands.npy"),  vc)
    np.save(os.path.join(out_dir, "test_users.npy"),  tu)
    np.save(os.path.join(out_dir, "test_truths.npy"), tt)
    np.save(os.path.join(out_dir, "test_cands.npy"),  tc)

    # ID maps + shapes (used by model/eval/app)
    with open(os.path.join(out_dir, "mappings.pkl"), "wb") as f:
        pickle.dump({"uids": uid_map, "iids": iid_map,
                     "num_users": num_users, "num_items": num_items}, f)

    return {"num_users": num_users, "num_items": num_items}