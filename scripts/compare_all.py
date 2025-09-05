from __future__ import annotations
import os, argparse, pandas as pd
from recsys.eval import eval_popularity, eval_itemknn, eval_neumf

def main():
    ap = argparse.ArgumentParser(description="Compare all models on val/test")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--Ks", type=str, default="5,10,20")
    ap.add_argument("--use_k_neighbors", type=int, default=50,
                    help="neighbors to use at inference for ItemKNN (<= max_neighbors)")
    ap.add_argument("--csv_out", default="compare_results.csv")
    args = ap.parse_args()

    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    rows = []

    for split in ["val", "test"]:
        for model in ["popularity", "itemknn", "neumf"]:
            for K in Ks:
                if model == "popularity":
                    hr, ndcg = eval_popularity(args.outputs, split, K)
                elif model == "itemknn":
                    hr, ndcg = eval_itemknn(args.outputs, split, K, args.use_k_neighbors)
                else:
                    hr, ndcg = eval_neumf(args.outputs, split, K)

                rows.append({
                    "Split": split.upper(),
                    "Model": f"{model}{'' if model!='itemknn' else f'(k={args.use_k_neighbors})'}",
                    "TopK": K,
                    "HR": hr,
                    "NDCG": ndcg,
                })

    df = pd.DataFrame(rows)

    print("\n=== Comparison Results ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    df.to_csv(args.csv_out, index=False)
    print(f"\n✅ Saved results to {args.csv_out}")

if __name__ == "__main__":
    main()
