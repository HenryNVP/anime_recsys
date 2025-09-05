# recsys/eval.py
from __future__ import annotations
import os, argparse, json, numpy as np, torch
from tqdm import tqdm
import pandas as pd

from .metrics import hr_ndcg_at_k
from .models.popularity import Popularity
from .models.itemknn import ItemKNN
from .models.neumf import NeuMF

def _load_splits(out_dir: str):
    train = pd.read_parquet(os.path.join(out_dir, "train.parquet"))
    val   = pd.read_parquet(os.path.join(out_dir, "val.parquet"))
    test  = pd.read_parquet(os.path.join(out_dir, "test.parquet"))
    return train, val, test

def eval_popularity(out_dir: str, split: str, k: int):
    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))
    pop = Popularity.load(out_dir)
    scores = np.vstack([pop.score_candidates(c) for c in cands])
    hr, ndcg = hr_ndcg_at_k(scores, k)
    return hr, ndcg

def eval_itemknn(out_dir: str, split: str, k: int, use_k_neighbors: int):
    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))
    train, _, _ = _load_splits(out_dir)
    user_seen = train.groupby("u")["i"].apply(lambda s: s.to_numpy(np.int64)).to_dict()

    knn = ItemKNN.load(out_dir)
    scores = np.zeros_like(cands, dtype=np.float32)
    for r in tqdm(range(cands.shape[0]), desc=f"Eval ItemKNN (use_k={use_k_neighbors})"):
        u = int(users[r]); cand = cands[r]
        seen = user_seen.get(u, np.empty(0, dtype=np.int64))
        scores[r] = knn.score_user_candidates(seen, cand, use_k=use_k_neighbors)
    return hr_ndcg_at_k(scores, k)

@torch.no_grad()
def eval_neumf(out_dir: str, split: str, k: int, emb_gmf=64, emb_mlp=64, mlp_layers=(256,128,64)):
    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))
    # with open(os.path.join(out_dir, "mappings.pkl"), "rb") as f:
    #     m = np.load.__self__.file = None  # silence type hints
    # load dims
    import pickle
    m = pickle.load(open(os.path.join(out_dir, "mappings.pkl"), "rb"))
    num_users, num_items = m["num_users"], m["num_items"]

    model = NeuMF(num_users, num_items, emb_gmf=emb_gmf, emb_mlp=emb_mlp, mlp_layers=mlp_layers)
    model.load_state_dict(torch.load(os.path.join(out_dir, "neumf.pt"), map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    scores = np.zeros_like(cands, dtype=float)
    for r in tqdm(range(cands.shape[0]), desc=f"Eval NeuMF ({split})"):
        u = int(users[r]); cand = cands[r]
        uu = torch.full((len(cand),), u, dtype=torch.long, device=device)
        ii = torch.tensor(cand, dtype=torch.long, device=device)
        scores[r] = model(uu, ii).detach().cpu().numpy()
    return hr_ndcg_at_k(scores, k)

def main():
    ap = argparse.ArgumentParser(description="Evaluate models (popularity, itemknn, neumf)")
    ap.add_argument("--model", required=True, choices=["popularity", "itemknn", "neumf"])
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--use_k_neighbors", type=int, default=50, help="for itemknn: neighbors to use at inference (<= max_neighbors)")
    args = ap.parse_args()

    if args.model == "popularity":
        hr, ndcg = eval_popularity(args.outputs, args.split, args.k)
    elif args.model == "itemknn":
        hr, ndcg = eval_itemknn(args.outputs, args.split, args.k, args.use_k_neighbors)
    else:
        hr, ndcg = eval_neumf(args.outputs, args.split, args.k)

    print(json.dumps({f"HR@{args.k}": hr, f"NDCG@{args.k}": ndcg}, indent=2))

if __name__ == "__main__":
    main()
