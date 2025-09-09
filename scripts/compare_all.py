# scripts/compare_all.py
from __future__ import annotations
import os, argparse, json
import numpy as np

from recsys.eval import eval_popularity, eval_itemknn, eval_neumf, eval_hybrid

def main():
    ap = argparse.ArgumentParser(description="Compare models on val/test for Ks")
    ap.add_argument("--data_dir", default="data_clean")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--Ks", type=str, default="5,10,20")
    ap.add_argument("--use_k_neighbors", type=int, default=50)  # for itemknn
    args = ap.parse_args()

    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    results = {k: {} for k in Ks}

    for K in Ks:
        print(f"\n=== K={K} on {args.split} ===")

        # Popularity
        hr, ndcg = eval_popularity(args.outputs, args.split, K)
        results[K]["popularity"] = (hr, ndcg)

        # ItemKNN
        hr, ndcg = eval_itemknn(args.outputs, args.data_dir, args.split, K, args.use_k_neighbors)
        results[K]["itemknn"] = (hr, ndcg)

        # NeuMF
        hr, ndcg = eval_neumf(args.outputs, args.data_dir, args.split, K)
        results[K]["neumf"] = (hr, ndcg)

        # Hybrid
        hr, ndcg = eval_hybrid(args.outputs, args.data_dir, args.split, K)
        results[K]["hybrid"] = (hr, ndcg)

    # pretty print
    print("\n=== Summary ===")
    models = ["popularity", "itemknn", "neumf", "hybrid"]
    for K in Ks:
        print(f"\nK={K}")
        for m in models:
            if m in results[K]:
                hr, ndcg = results[K][m]
                print(f"  {m:12s}  HR@{K}={hr:.4f}  NDCG@{K}={ndcg:.4f}")

    # optional: save JSON
    out_json = os.path.join(args.outputs, f"compare_{args.split}.json")
    with open(out_json, "w") as f:
        json.dump({str(k): {m: {"HR": float(v[0]), "NDCG": float(v[1])} for m, v in results[k].items()} for k in Ks},
                  f, indent=2)
    print(f"\nSaved: {out_json}")

if __name__ == "__main__":
    main()
