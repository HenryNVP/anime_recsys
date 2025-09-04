# recsys/eval.py
from __future__ import annotations
import os, argparse, json
import numpy as np
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix

from .models.neumf import NeuMF
from .models.knn import UserKNN


# ---------- shared metric helper ----------
def hr_ndcg_at_k(scores: np.ndarray, k: int = 10):
    """
    Given scores (rows = users, cols = candidate items),
    where col0 in each row is the TRUE item,
    compute HR@k and NDCG@k.
    """
    order = np.argsort(-scores, axis=1)         # descending
    topk = order[:, :k]
    hits = (topk == 0).any(axis=1).astype(float)

    pos = np.argwhere(order == 0)               # where truth appears
    ranks = np.full(order.shape[0], order.shape[1], dtype=int)
    ranks[pos[:, 0]] = pos[:, 1]

    ndcg = np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0)
    return hits.mean(), ndcg.mean()


# ---------- NeuMF eval ----------
@torch.no_grad()
def eval_neumf(out_dir: str, split: str = "test", k: int = 10, emb_gmf=16, emb_mlp=32, mlp_layers=(256,128,64)):
    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))

    meta = json.load(open(os.path.join(out_dir, "mappings.pkl"), "rb"))
    num_users, num_items = meta["num_users"], meta["num_items"]

    # rebuild model with correct dims
    model = NeuMF(num_users, num_items, emb_gmf=emb_gmf, emb_mlp=emb_mlp, mlp_layers=mlp_layers)
    model.load_state_dict(torch.load(os.path.join(out_dir, "model.pt"), map_location="cpu"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scores = np.zeros_like(cands, dtype=float)
    for r in tqdm(range(cands.shape[0]), desc=f"Eval NeuMF ({split})"):
        u = int(users[r]); cand = cands[r]
        uu = torch.full((len(cand),), u, dtype=torch.long, device=device)
        ii = torch.tensor(cand, dtype=torch.long, device=device)
        s = model(uu, ii).detach().cpu().numpy()
        scores[r] = s

    hr, ndcg = hr_ndcg_at_k(scores, k=k)
    return hr, ndcg


# ---------- UserKNN eval ----------
def eval_userknn(out_dir: str, split: str = "test", k: int = 10):
    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))

    pack = np.load(os.path.join(out_dir, "userknn_ui.npz"), allow_pickle=False)
    ui = csr_matrix((pack["data"], pack["indices"], pack["indptr"]), shape=tuple(pack["shape"]))
    norms = pack["norms"]

    with open(os.path.join(out_dir, "userknn_meta.json")) as f:
        meta = json.load(f)
    k_neigh = meta.get("k_neighbors", 50)

    model = UserKNN(k_neighbors=k_neigh)
    model.ui = ui
    model.user_norms = norms

    scores = np.zeros_like(cands, dtype=float)
    for r in tqdm(range(cands.shape[0]), desc=f"Eval UserKNN ({split})"):
        scores[r] = model.score_user_candidates(int(users[r]), cands[r])

    hr, ndcg = hr_ndcg_at_k(scores, k=k)
    return hr, ndcg


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Evaluate models on Anime dataset")
    ap.add_argument("--model", type=str, required=True, choices=["neumf", "userknn"])
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--outputs", type=str, default="outputs")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    if args.model == "neumf":
        hr, ndcg = eval_neumf(args.outputs, split=args.split, k=args.k)
    elif args.model == "userknn":
        hr, ndcg = eval_userknn(args.outputs, split=args.split, k=args.k)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    print(json.dumps({f"HR@{args.k}": hr, f"NDCG@{args.k}": ndcg}, indent=2))


if __name__ == "__main__":
    main()
