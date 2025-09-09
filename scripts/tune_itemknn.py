# scripts/tune_itemknn.py
from __future__ import annotations
import os, argparse, numpy as np, pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from recsys.metrics import hr_ndcg_at_k
from recsys.models.itemknn import ItemKNN

def main():
    ap = argparse.ArgumentParser(description="Tune ItemKNN neighbors on validation set")
    ap.add_argument("--outputs", default="outputs", help="Folder with saved models/splits")
    ap.add_argument("--split", default="val", choices=["val","test"],
                    help="Which split to evaluate (val for tuning, test for final report)")
    ap.add_argument("--eval_k", type=int, default=10, help="Top-K cutoff for HR/NDCG")
    ap.add_argument("--max_k", type=int, default=200, help="Max neighbors to try")
    ap.add_argument("--step", type=int, default=10, help="Step size for neighbors sweep")
    ap.add_argument("--csv_out", default="itemknn_tuning.csv")
    args = ap.parse_args()

    # Load candidate arrays
    users = np.load(os.path.join(args.outputs, f"{args.split}_users.npy"))
    cands = np.load(os.path.join(args.outputs, f"{args.split}_cands.npy"))

    # Load TRAIN to build per-user seen items
    train_df = pd.read_parquet(os.path.join(args.outputs, "train.parquet"))
    user_seen = train_df.groupby("u")["i"].apply(lambda s: s.to_numpy(np.int64)).to_dict()

    # Load ItemKNN neighbors
    knn = ItemKNN.load(args.outputs)

    ks = list(range(args.step, args.max_k+1, args.step))
    rows = []
    for use_k in ks:
        scores = np.zeros_like(cands, dtype=np.float32)
        for r in tqdm(range(cands.shape[0]), desc=f"[{args.split}] use_k={use_k}"):
            u = int(users[r]); cand = cands[r]
            seen = user_seen.get(u, np.empty(0, dtype=np.int64))
            scores[r] = knn.score_user_candidates(seen, cand, use_k=use_k)
        hr, ndcg = hr_ndcg_at_k(scores, k=args.eval_k)
        rows.append({"use_k": use_k, f"HR@{args.eval_k}": hr, f"NDCG@{args.eval_k}": ndcg})

    df = pd.DataFrame(rows)
    print("\n=== ItemKNN Tuning Results ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    df.to_csv(args.csv_out, index=False)
    print(f"\n✅ Saved results to {args.csv_out}")

    # Plot curves
    plt.figure()
    plt.plot(df["use_k"], df[f"HR@{args.eval_k}"], marker="o", label=f"HR@{args.eval_k}")
    plt.plot(df["use_k"], df[f"NDCG@{args.eval_k}"], marker="o", label=f"NDCG@{args.eval_k}")
    plt.xlabel("ItemKNN neighbors (use_k)")
    plt.ylabel("Metric")
    plt.title(f"ItemKNN {args.split.upper()} sweep (Top-{args.eval_k})")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
