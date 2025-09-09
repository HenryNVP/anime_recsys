import os, argparse, csv
from recsys.eval import eval_itemknn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_clean")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--split", default="val", choices=["val","test"])
    ap.add_argument("--eval_k", type=int, default=10)
    ap.add_argument("--max_k", type=int, default=200)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--csv_out", default="itemknn_tuning.csv")
    args = ap.parse_args()

    # ensure required artifacts exist (produced by: python -m recsys.train --trainer itemknn ...)
    for fn in ["itemknn_topk.npz", "ui_csr.npz", f"{args.split}_users.npy", f"{args.split}_cands.npy"]:
        path = os.path.join(args.outputs, fn)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Train ItemKNN & build candidates first.")

    rows = [("use_k", f"HR@{args.eval_k}", f"NDCG@{args.eval_k}")]
    for use_k in range(args.step, args.max_k + 1, args.step):
        print(f"\nuse_k_neighbors={use_k}")
        hr, ndcg = eval_itemknn(args.outputs, args.data_dir, args.split, args.eval_k, use_k_neighbors=use_k)
        rows.append((use_k, hr, ndcg))

    with open(args.csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\nSaved: {args.csv_out}")

if __name__ == "__main__":
    main()