import os, argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data")          # where your full csvs live
    ap.add_argument("--dst_dir", default="data_mini")     # where to write the mini set
    ap.add_argument("--users", type=int, default=300)     # keep first N users
    ap.add_argument("--items", type=int, default=800)     # keep first M items (after filtering)
    args = ap.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)

    # load full
    ratings = pd.read_csv(os.path.join(args.src_dir, "rating.csv"))
    anime   = pd.read_csv(os.path.join(args.src_dir, "anime.csv"))

    # keep explicit positives
    ratings = ratings[ratings["rating"] != -1].copy()

    # pick first N distinct users
    keep_users = ratings["user_id"].drop_duplicates().head(args.users).tolist()
    mini = ratings[ratings["user_id"].isin(keep_users)].copy()

    # within these, pick top-M items by frequency
    top_items = (
        mini["anime_id"]
        .value_counts()
        .head(args.items)
        .index
        .tolist()
    )
    mini = mini[mini["anime_id"].isin(top_items)].copy()

    # write mini rating.csv
    mini.to_csv(os.path.join(args.dst_dir, "rating.csv"), index=False)

    # write mini anime.csv (only rows for items we kept)
    anime_mini = anime[anime["anime_id"].isin(top_items)].copy()
    anime_mini.to_csv(os.path.join(args.dst_dir, "anime.csv"), index=False)

    # tiny report
    u = mini["user_id"].nunique()
    i = mini["anime_id"].nunique()
    n = len(mini)
    print(f"✅ Wrote {args.dst_dir}/ with {u} users, {i} items, {n} interactions.")

if __name__ == "__main__":
    main()