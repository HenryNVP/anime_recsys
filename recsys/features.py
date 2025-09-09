# recsys/features.py
from __future__ import annotations
import os, json, html, pickle
import numpy as np
import pandas as pd

def _split_genres(s: str) -> list[str]:
    if not isinstance(s, str) or not s:
        return []
    s = html.unescape(s)
    return [g.strip() for g in s.split(",") if g.strip()]

def _minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(np.nanmin(x)), float(np.nanmax(x))
    denom = (mx - mn) if (mx > mn) else 1.0
    return (x - mn) / denom

def make_item_features(data_dir: str = "data_clean", out_dir: str = "outputs"):
    """
    Build item feature matrix aligned to outputs/mappings.pkl (created by preprocess.py).
    Saves:
      - outputs/item_feats.npy        (float32, [num_items, F])
      - outputs/item_feats_meta.json  (vocab + dims + scaling notes)
    """
    os.makedirs(out_dir, exist_ok=True)

    # load mappings (reindexed item ids)
    with open(os.path.join(data_dir, "mappings.pkl"), "rb") as f:
        M = pickle.load(f)
    iid_map: dict = M["iid_map"]
    num_items: int = M["num_items"]

    # load cleaned anime metadata (already deduped & episodes numeric)
    anime = pd.read_csv(os.path.join(data_dir, "anime.csv"))

    # ---------- vocabularies ----------
    # Genres (multi-hot)
    genre_set = set()
    for g in anime["genre"].astype(str).fillna(""):
        genre_set.update(_split_genres(g))
    genre_vocab = sorted(genre_set)
    g2idx = {g:i for i,g in enumerate(genre_vocab)}
    G = len(genre_vocab)

    # Type (one-hot with <UNK>)
    type_vals = anime["type"].astype(str).replace({"nan": ""})
    type_unique = sorted([t for t in type_vals.dropna().unique().tolist() if t])
    type_vocab = type_unique + ["<UNK>"]
    t2idx = {t:i for i,t in enumerate(type_vocab)}
    T = len(type_vocab)

    # ---------- numeric features (with scaling & missing handling) ----------
    # Episodes: already coerced to numeric in preprocess; may contain NaN.
    ep_series = anime["episodes"].astype(float)
    ep_raw = np.log1p(ep_series.fillna(0).clip(lower=0).to_numpy(np.float64))
    ep_scaled = _minmax(ep_raw)

    # Rating: min-max [0,10]; impute median (column-wise)
    rating_col = anime["rating"].astype(float)
    rating_missing = rating_col.isna().to_numpy()
    rating_impute = float(np.nanmedian(rating_col.to_numpy()))
    rating_filled = rating_col.fillna(rating_impute).clip(lower=0, upper=10).to_numpy(np.float64)
    rating_scaled = (rating_filled / 10.0).astype(np.float32)

    # Members: log1p + minmax
    mem_raw = np.log1p(anime["members"].fillna(0).clip(lower=0).to_numpy(np.float64))
    mem_scaled = _minmax(mem_raw)

    # ---------- feature matrix layout ----------
    # [ genres (G) | type (T) | episodes (1) | rating (1) | members (1) | miss_genre(1) | miss_type(1) | miss_rating(1) ]
    F = G + T + 3 + 3
    feats = np.zeros((num_items, F), dtype=np.float32)

    # Missing masks
    genre_missing = anime["genre"].isna() | (anime["genre"].astype(str).str.strip() == "")
    type_missing  = anime["type"].isna()  | (anime["type"].astype(str).str.strip() == "")

    # Fill categorical blocks
    for r in anime.itertuples(index=False):
        aid = int(getattr(r, "anime_id"))
        j = iid_map.get(aid, None)
        if j is None:
            continue

        # genres multi-hot
        gs = _split_genres(getattr(r, "genre")) if not pd.isna(getattr(r, "genre")) else []
        for g in gs:
            idx = g2idx.get(g, None)
            if idx is not None:
                feats[j, idx] = 1.0

        # type one-hot with UNK
        tval = str(getattr(r, "type")) if not pd.isna(getattr(r, "type")) else ""
        t_idx = t2idx["<UNK>"] if (tval == "") else t2idx.get(tval, t2idx["<UNK>"])
        feats[j, G + t_idx] = 1.0

    # numeric slices
    base_num = G + T
    row_idx = anime["anime_id"].map(iid_map).dropna().astype(int).to_numpy()
    feats[row_idx, base_num + 0] = ep_scaled[row_idx]
    feats[row_idx, base_num + 1] = rating_scaled[row_idx]
    feats[row_idx, base_num + 2] = mem_scaled[row_idx]

    # missingness indicators
    base_miss = base_num + 3
    feats[row_idx, base_miss + 0] = genre_missing.loc[anime["anime_id"].isin(iid_map.keys())].to_numpy(dtype=np.float32)
    feats[row_idx, base_miss + 1] = type_missing.loc[anime["anime_id"].isin(iid_map.keys())].to_numpy(dtype=np.float32)
    feats[row_idx, base_miss + 2] = rating_missing[anime["anime_id"].isin(iid_map.keys())].astype(np.float32)

    # ---------- save ----------
    np.save(os.path.join(out_dir, "item_feats.npy"), feats)
    meta = {
        "genre_vocab": genre_vocab,
        "type_vocab": type_vocab,
        "dims": {"genres": G, "types": T, "numeric": 3, "missing": 3, "total": int(F)},
        "scaling": {
            "episodes": "log1p + minmax (column-wise after fillna(0))",
            "rating":   "minmax on [0,10], impute=median + missing flag",
            "members":  "log1p + minmax (column-wise after fillna(0))",
        },
        "notes": "HTML entities unescaped in genres; <UNK> type bucket; missingness flags added.",
    }
    with open(os.path.join(out_dir, "item_feats_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Saved item features: {feats.shape} → {os.path.join(out_dir, 'item_feats.npy')}")
