# scripts/preprocess.py
from __future__ import annotations
import os, argparse, pickle
import numpy as np
import pandas as pd

def split_one_seed(ratings: pd.DataFrame, min_user=5, min_item=5, seed=42):
    ratings = ratings[ratings["rating"] != -1].copy()

    # filter sparsity
    uc = ratings["user_id"].value_counts()
    ic = ratings["anime_id"].value_counts()
    ratings = ratings[
        ratings["user_id"].isin(uc[uc >= min_user].index) &
        ratings["anime_id"].isin(ic[ic >= min_item].index)
    ].copy()

    # reindex
    uid_map = {u:i for i,u in enumerate(ratings["user_id"].unique())}
    iid_map = {a:j for j,a in enumerate(ratings["anime_id"].unique())}
    ratings["u"] = ratings["user_id"].map(uid_map).astype("int64")
    ratings["i"] = ratings["anime_id"].map(iid_map).astype("int64")

    num_users = ratings["u"].nunique()
    num_items = ratings["i"].nunique()

    # per-user 70/15/15
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
    mappings = {"num_users": num_users, "num_items": num_items, "uid_map": uid_map, "iid_map": iid_map}
    return train, val, test, mappings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data_raw")
    ap.add_argument("--out_root", default="data_clean")      # root folder
    ap.add_argument("--seeds", default="42")                 # "42" or "42,43,44"
    ap.add_argument("--min_user", type=int, default=5)
    ap.add_argument("--min_item", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    anime   = pd.read_csv(os.path.join(args.raw_dir, "anime.csv"))
    ratings = pd.read_csv(os.path.join(args.raw_dir, "rating.csv"))

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    for seed in seeds:
        out_dir = os.path.join(args.out_root, f"seed{seed}")
        os.makedirs(out_dir, exist_ok=True)
        train, val, test, m = split_one_seed(ratings, args.min_user, args.min_item, seed)
        train.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
        val.to_parquet(  os.path.join(out_dir, "val.parquet"),   index=False)
        test.to_parquet( os.path.join(out_dir, "test.parquet"),  index=False)
        with open(os.path.join(out_dir, "mappings.pkl"), "wb") as f:
            pickle.dump(m, f)
        print(f"✅ Wrote splits for seed {seed} to {out_dir}")

if __name__ == "__main__":
    main()
