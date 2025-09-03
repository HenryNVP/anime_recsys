import os, argparse, pickle, json
import numpy as np
import torch

from .metrics import hr_ndcg_at_k
from .models.neumf import NeuMF

def _load_shapes(out_dir: str):
    with open(os.path.join(out_dir, "mappings.pkl"), "rb") as f:
        m = pickle.load(f)
    return m["num_users"], m["num_items"]

def eval_neumf(out_dir: str, split: str = "val", k: int = 10):
    num_users, num_items = _load_shapes(out_dir)
    model = NeuMF(num_users, num_items)
    model.load_state_dict(torch.load(os.path.join(out_dir, "model.pt"), map_location="cpu"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))

    scores = np.zeros_like(cands, dtype=float)
    with torch.no_grad():
        for r in range(cands.shape[0]):
            u = int(users[r]); cand = cands[r]
            uu = torch.full((len(cand),), u, dtype=torch.long, device=device)
            ii = torch.tensor(cand, dtype=torch.long, device=device)
            scores[r] = model(uu, ii).detach().cpu().numpy()

    return hr_ndcg_at_k(scores, k=k)

def eval_als(out_dir: str, split: str = "val", k: int = 10):
    # Load factors produced by train_als
    user_f = np.load(os.path.join(out_dir, "als_user_factors.npy"))
    item_f = np.load(os.path.join(out_dir, "als_item_factors.npy"))

    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))

    scores = np.zeros_like(cands, dtype=float)
    for r in range(cands.shape[0]):
        u = int(users[r]); cand = cands[r]
        scores[r] = item_f[cand] @ user_f[u]
    return hr_ndcg_at_k(scores, k=k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=str, default="outputs")
    ap.add_argument("--model", type=str, default="neumf", choices=["neumf", "als"])
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    if args.model == "neumf":
        hr, ndcg = eval_neumf(args.outputs, args.split, args.k)
    else:
        hr, ndcg = eval_als(args.outputs, args.split, args.k)

    os.makedirs("results", exist_ok=True)
    out = {"model": args.model, "split": args.split, f"HR@{args.k}": hr, f"NDCG@{args.k}": ndcg}
    with open("results/metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
