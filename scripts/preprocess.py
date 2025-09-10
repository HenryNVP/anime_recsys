# scripts/preprocess.py
from __future__ import annotations
import os, argparse, json, pickle
from typing import Tuple
import numpy as np
import pandas as pd

def _per_user_split(df: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    train = pd.concat([p[0] for p in parts], ignore_index=True)[["u","i"]]
    val   = pd.concat([p[1] for p in parts], ignore_index=True)[["u","i"]]
    test  = pd.concat([p[2] for p in parts], ignore_index=True)[["u","i"]]
    return train, val, test

def main():
    ap = argparse.ArgumentParser(description="Preprocess Anime dataset: clean -> de-dup -> reindex -> 70/15/15 splits")
    ap.add_argument("--raw_dir",   type=str, default="data_raw",   help="folder with original anime.csv and rating.csv")
    ap.add_argument("--clean_dir", type=str, default="data_clean", help="output folder for cleaned data + splits")
    ap.add_argument("--min_user_inter", type=int, default=5, help="min interactions per user")
    ap.add_argument("--min_item_inter", type=int, default=5, help="min interactions per item")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio",   type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs(args.clean_dir, exist_ok=True)

    # ---------- Load raw ----------
    anime_csv   = os.path.join(args.raw_dir, "anime.csv")
    rating_csv  = os.path.join(args.raw_dir, "rating.csv")
    assert os.path.exists(anime_csv),  f"Missing {anime_csv}"
    assert os.path.exists(rating_csv), f"Missing {rating_csv}"

    anime   = pd.read_csv(anime_csv)
    ratings = pd.read_csv(rating_csv)

    # ---------- Clean: coerce 'episodes' numeric; drop duplicates ----------
    # (coerce strings like "Unknown" → NaN, keep NaN; features.py will handle later)
    if "episodes" in anime.columns:
        anime["episodes"] = pd.to_numeric(anime["episodes"], errors="coerce")

    # Drop exact duplicate rows first
    anime   = anime.drop_duplicates()
    ratings = ratings.drop_duplicates()

    # Then deduplicate on primary keys if necessary
    if "anime_id" in anime.columns:
        anime = anime.sort_values(by=list(anime.columns)).drop_duplicates(subset=["anime_id"], keep="last")
    ratings = ratings.sort_values(by=list(ratings.columns)).drop_duplicates(subset=["user_id","anime_id"], keep="last")

    # Remove '-1' sentinel from interactions (implicit positives only)
    if "rating" in ratings.columns:
        ratings = ratings[ratings["rating"] != -1].copy()

    # ---------- Filter sparsity ----------
    min_user = 5
    min_item = 5
    uc = ratings["user_id"].value_counts()
    ic = ratings["anime_id"].value_counts()
    ratings = ratings[
        ratings["user_id"].isin(uc[uc >= min_user].index) &
        ratings["anime_id"].isin(ic[ic >= min_item].index)
    ].copy()
    interactions = len(ratings)

    # ---------- Reindex to contiguous ids ----------
    uid_map = {u: i for i, u in enumerate(ratings["user_id"].unique())}
    iid_map = {a: j for j, a in enumerate(ratings["anime_id"].unique())}
    ratings["u"] = ratings["user_id"].map(uid_map).astype("int64")
    ratings["i"] = ratings["anime_id"].map(iid_map).astype("int64")

    num_users = int(ratings["u"].nunique())
    num_items = int(ratings["i"].nunique())

    # Align anime table to kept items only
    if "anime_id" in anime.columns:
        anime_keep = anime[anime["anime_id"].isin(iid_map.keys())].copy()
    else:
        anime_keep = anime.copy()

    # ---------- Splits (canonical 70/15/15) ----------
    train_df, val_df, test_df = _per_user_split(
        ratings[["u","i"]], seed=args.seed,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    # ---------- Save cleaned data ----------
    # Clean CSVs (with reindexed columns included in ratings)
    anime_keep.to_csv(os.path.join(args.clean_dir, "anime.csv"), index=False)
    ratings.to_csv(os.path.join(args.clean_dir, "rating.csv"), index=False)

    # Mappings + stats
    mappings = {"num_users": num_users, "num_items": num_items, "uid_map": uid_map, "iid_map": iid_map}
    with open(os.path.join(args.clean_dir, "mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)

    stats = {
        "interactions": interactions,
        "num_users": num_users, "num_items": num_items,
        "min_user_inter": args.min_user_inter, "min_item_inter": args.min_item_inter,
        "train_ratio": args.train_ratio, "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio, "seed": args.seed,
        "notes": "episodes coerced to numeric (NaN kept), duplicates removed",
    }
    json.dump(stats, open(os.path.join(args.clean_dir, "stats.json"), "w"), indent=2)

    # Splits (parquet)
    train_df.to_parquet(os.path.join(args.clean_dir, "train.parquet"), index=False)
    val_df.to_parquet(  os.path.join(args.clean_dir, "val.parquet"),   index=False)
    test_df.to_parquet( os.path.join(args.clean_dir, "test.parquet"),  index=False)

    # ---------- Summary ----------
    print("\n=== Preprocess Summary ===")
    print(f"Interactions:    {interactions:,}")
    print(f"Users (kept):     {num_users:,}")
    print(f"Items (kept):     {num_items:,}")
    print(f"Clean folder:     {args.clean_dir}")
    print(f"- train.parquet   {len(train_df):,}")
    print(f"- val.parquet     {len(val_df):,}")
    print(f"- test.parquet    {len(test_df):,}")
    print("Files: anime.csv, rating.csv, mappings.pkl, stats.json")
    print("Done ✅")

if __name__ == "__main__":
    main()
