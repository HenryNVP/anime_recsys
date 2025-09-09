import os, json
import pandas as pd
import numpy as np

def check_features(clean_dir="data_clean", outputs="outputs"):
    anime = pd.read_csv(os.path.join(clean_dir, "anime.csv"))
    with open(os.path.join(outputs, "item_feats_meta.json")) as f:
        meta = json.load(f)
    feats = np.load(os.path.join(outputs, "item_feats.npy"))

    print("=== Anime metadata (cleaned) ===")
    print(f"Total items: {len(anime)}")
    print(f"Feature matrix shape: {feats.shape}")

    # --- Missingness stats ---
    miss_genre = anime["genre"].isna().sum()
    miss_type  = anime["type"].isna().sum()
    miss_rating = anime["rating"].isna().sum()

    print("\n=== Missing values in anime.csv ===")
    print(f"Missing genres:  {miss_genre} ({miss_genre/len(anime):.2%})")
    print(f"Missing types:   {miss_type} ({miss_type/len(anime):.2%})")
    print(f"Missing ratings: {miss_rating} ({miss_rating/len(anime):.2%})")

    # --- Numeric summaries ---
    print("\n=== Numeric fields summary (before scaling) ===")
    for col in ["episodes", "rating", "members"]:
        desc = anime[col].describe(percentiles=[.25, .5, .75])
        print(f"\n{col}:")
        print(desc)

    # --- Feature matrix sanity ---
    feat_cols = (
        [f"genre_{g}" for g in meta["genre_vocab"]] +
        [f"type_{t}" for t in meta["type_vocab"]] +
        ["episodes", "rating", "members",
         "miss_genre", "miss_type", "miss_rating"]
    )

    df_feats = pd.DataFrame(feats, columns=feat_cols)

    print("\n=== Sanity check: missingness flags ===")
    for c in ["miss_genre", "miss_type", "miss_rating"]:
        print(f"{c}: {int(df_feats[c].sum())} items flagged ({df_feats[c].mean():.2%})")

    print("\n=== Sanity check: numeric feature ranges (scaled) ===")
    for c in ["episodes", "rating", "members"]:
        print(f"{c}: min={df_feats[c].min():.3f}, max={df_feats[c].max():.3f}, mean={df_feats[c].mean():.3f}")

if __name__ == "__main__":
    check_features()
