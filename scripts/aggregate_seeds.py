# scripts/aggregate_seeds.py
from __future__ import annotations
import os, json, argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_root", default="outputs")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--split", default="test", choices=["val","test"])
    args = ap.parse_args()

    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    acc = {}  # acc[(model, K)] -> list[(HR, NDCG)]

    for s in seeds:
        path = os.path.join(args.outputs_root, f"seed{s}", f"compare_{args.split}.json")
        if not os.path.exists(path):
            print(f"Missing: {path} (skipping)")
            continue
        blob = json.load(open(path))
        for K, models in blob.items():
            for m, vals in models.items():
                acc.setdefault((m, int(K)), []).append((vals["HR"], vals["NDCG"]))

    # print summary
    print(f"\n=== Aggregate over seeds {seeds} ({args.split}) ===")
    keys = sorted(acc.keys(), key=lambda x: (x[1], x[0]))
    for (m, K) in keys:
        arr = np.asarray(acc[(m,K)], dtype=float)  # n x 2
        if len(arr) == 0: continue
        hr_mean, hr_std = arr[:,0].mean(), arr[:,0].std()
        nd_mean, nd_std = arr[:,1].mean(), arr[:,1].std()
        print(f"K={K:2d}  {m:12s}  HR={hr_mean:.4f}±{hr_std:.4f}  NDCG={nd_mean:.4f}±{nd_std:.4f}")

if __name__ == "__main__":
    main()
