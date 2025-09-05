# mini_cf_demo.py
import os
import numpy as np
import pandas as pd

# --------------------- helpers ---------------------
def per_user_split(df, seed=0, train_ratio=0.70, val_ratio=0.15):
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby("u", sort=False):
        idx = np.arange(len(g)); rng.shuffle(idx)
        n = len(g)
        n_tr = max(1, int(train_ratio * n))
        n_va = max(1, int(val_ratio * n))
        tr = g.iloc[idx[:n_tr]]
        va = g.iloc[idx[n_tr:n_tr+n_va]]
        te = g.iloc[idx[n_tr+n_va:]]
        parts.append((tr, va, te))
    train = pd.concat([p[0] for p in parts], ignore_index=True)
    val   = pd.concat([p[1] for p in parts], ignore_index=True)
    test  = pd.concat([p[2] for p in parts], ignore_index=True)
    return train, val, test

def user_pos_map(df):
    return df.groupby("u")["i"].apply(set).to_dict()

def build_eval_candidates(train_df, split_df, num_items, num_negs=20, seed=1):
    rng = np.random.default_rng(seed)
    pos = user_pos_map(train_df)
    all_items = np.arange(num_items, dtype=np.int64)

    users = split_df["u"].to_numpy(np.int64)
    truths = split_df["i"].to_numpy(np.int64)
    cands = []
    for u, t in zip(users, truths):
        seen = pos.get(int(u), set())
        seen_arr = np.fromiter(seen, dtype=np.int64) if seen else np.empty(0, dtype=np.int64)
        pool = np.setdiff1d(all_items, seen_arr, assume_unique=True)
        k = min(num_negs, len(pool))
        negs = rng.choice(pool, size=k, replace=False) if k > 0 else rng.choice(all_items, size=num_negs, replace=True)
        cands.append(np.concatenate([[t], negs]))
    return users, truths, np.stack(cands, axis=0)

def hr_ndcg(scores, K):
    # scores: [N, C], column 0 is the TRUE item
    order = np.argsort(-scores, axis=1)
    topk = order[:, :K]
    hits = (topk == 0).any(axis=1).astype(float)

    pos = np.argwhere(order == 0)   # (row, col) where truth appears
    ranks = np.full(order.shape[0], order.shape[1], dtype=int)
    ranks[pos[:,0]] = pos[:,1]
    ndcg = np.where(ranks < K, 1.0 / np.log2(ranks + 2), 0.0)
    return hits.mean(), ndcg.mean()

# --------------------- minimal pipeline ---------------------
def main(
    ratings_csv="data/rating.csv",
    anime_csv="data/anime.csv",
    keep_users=300,      # tiny subset for speed
    keep_items=800,      # tiny subset for speed
    seed=42,
    neigh_k=50,          # KNN neighbors (hyperparam)
    eval_ks=(5, 10, 20)  # Top-K cutoff for metrics
):
    # 1) Load and thin the data (tiny subset)
    assert os.path.exists(ratings_csv), f"Missing {ratings_csv}"
    assert os.path.exists(anime_csv), f"Missing {anime_csv}"
    ratings = pd.read_csv(ratings_csv)
    ratings = ratings[ratings["rating"] != -1].copy()  # keep explicit interactions

    # keep first N users
    keep_u = ratings["user_id"].drop_duplicates().head(keep_users).tolist()
    mini = ratings[ratings["user_id"].isin(keep_u)].copy()
    # keep M most frequent items within these
    keep_i = mini["anime_id"].value_counts().head(keep_items).index.tolist()
    mini = mini[mini["anime_id"].isin(keep_i)].copy()

    # 2) Reindex to contiguous [0..U), [0..I)
    uid_map = {u: idx for idx, u in enumerate(mini["user_id"].unique())}
    iid_map = {a: idx for idx, a in enumerate(mini["anime_id"].unique())}
    mini["u"] = mini["user_id"].map(uid_map).astype("int64")
    mini["i"] = mini["anime_id"].map(iid_map).astype("int64")

    num_users = mini["u"].nunique()
    num_items = mini["i"].nunique()
    print(f"[mini] users={num_users}, items={num_items}, interactions={len(mini)}")

    # 3) Per-user split
    train_df, val_df, test_df = per_user_split(mini[["u","i"]], seed=seed)

    # 4) Build candidate lists (true + 20 negs) for val/test
    val_users, _, val_cands = build_eval_candidates(train_df, val_df, num_items, num_negs=20, seed=seed+1)
    test_users, _, test_cands = build_eval_candidates(train_df, test_df, num_items, num_negs=20, seed=seed+2)

    # 5) Build dense UI from TRAIN only (tiny = OK)
    UI = np.zeros((num_users, num_items), dtype=np.float32)
    UI[train_df["u"], train_df["i"]] = 1.0

    # 6) User-user cosine (dense, tiny safe)
    # Normalize rows: u_norm = u / ||u||
    row_norms = np.linalg.norm(UI, axis=1, keepdims=True) + 1e-12
    UI_norm = UI / row_norms
    # Cosine sim = UI_norm @ UI_norm.T
    user_sim = UI_norm @ UI_norm.T
    np.fill_diagonal(user_sim, 0.0)  # remove self

    # 7) Scoring function: aggregate neighbors' preferences for candidates
    def score_user_candidates(u, cand, k_neighbors=neigh_k):
        sim = user_sim[u]
        k = min(k_neighbors, sim.size)
        # top-k neighbor indices
        if k > 0:
            topk_idx = np.argpartition(-sim, k-1)[:k]
            topk_sim = sim[topk_idx]                          # (k,)
            neigh_inter = UI[topk_idx][:, cand]               # (k, |C|)
            return topk_sim @ neigh_inter                     # (|C|,)
        else:
            return np.zeros(len(cand), dtype=np.float32)

    # 8) Evaluate on val and test
    def eval_split(users, cands, Ks):
        scores = np.zeros_like(cands, dtype=np.float32)
        for r in range(cands.shape[0]):
            scores[r] = score_user_candidates(int(users[r]), cands[r], neigh_k)
        out = {}
        for K in Ks:
            hr, ndcg = hr_ndcg(scores, K)
            out[K] = (hr, ndcg)
        return out

    print("\n[VAL]")
    val_results = eval_split(val_users, val_cands, eval_ks)
    for K, (hr, ndcg) in val_results.items():
        print(f"  HR@{K}: {hr:.4f}   NDCG@{K}: {ndcg:.4f}")

    print("\n[TEST]")
    test_results = eval_split(test_users, test_cands, eval_ks)
    for K, (hr, ndcg) in test_results.items():
        print(f"  HR@{K}: {hr:.4f}   NDCG@{K}: {ndcg:.4f}")

if __name__ == "__main__":
    main()
