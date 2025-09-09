# recsys/eval.py
from __future__ import annotations
import os, argparse, pickle
import numpy as np
import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix

from .metrics import hr_ndcg_at_k
from .models.neumf import NeuMF
from .models.hybrid_neumf import HybridNeuMF


def _load_candidates(outputs: str, split: str):
    users = np.load(os.path.join(outputs, f"{split}_users.npy"))
    truths = np.load(os.path.join(outputs, f"{split}_truths.npy"))
    cands = np.load(os.path.join(outputs, f"{split}_cands.npy"))
    return users, truths, cands


# ---------- Popularity ----------

def eval_popularity(outputs: str, split: str, k: int):
    users, truths, cands = _load_candidates(outputs, split)
    pop = np.load(os.path.join(outputs, "popularity.npy"))
    scores = pop[cands]  # broadcast: (num_rows, num_cands)
    hr, ndcg = hr_ndcg_at_k(scores, k)
    print(f"HR@{k}={hr:.4f}  NDCG@{k}={ndcg:.4f}")
    return hr, ndcg


# ---------- ItemKNN ----------

def _load_sparse(path):
    pack = np.load(path, allow_pickle=True)
    return csr_matrix((pack["data"], pack["indices"], pack["indptr"]), shape=tuple(pack["shape"]))

def eval_itemknn(outputs: str, data_dir: str, split: str, k: int, use_k_neighbors: int = 50):
    users, truths, cands = _load_candidates(outputs, split)
    # user history
    ui = _load_sparse(os.path.join(outputs, "ui_csr.npz"))  # (U, I)
    # item-item top-k neighbor list (we’ll use only top use_k per item)
    sim = _load_sparse(os.path.join(outputs, "itemknn_topk.npz")).tocsr()  # (I, I)

    # score(u, j) = sum_{i in user u history} sim[i, j]
    scores = np.zeros_like(cands, dtype=np.float32)
    for r in tqdm(range(cands.shape[0]), desc=f"Eval ItemKNN ({split})"):
        u = int(users[r])
        hist = ui.getrow(u)  # 1 x I
        # pre-compute a sparse vector: s = hist * sim  => (1 x I)
        # but we only need entries at cands[r]
        s = hist @ sim  # (1 x I) csr
        row_scores = s[:, cands[r]].toarray().ravel()
        # zero self-sim if candidate is in history (safety)
        scores[r] = row_scores

    hr, ndcg = hr_ndcg_at_k(scores, k)
    print(f"HR@{k}={hr:.4f}  NDCG@{k}={ndcg:.4f}")
    return hr, ndcg


# ---------- NeuMF ----------

@torch.no_grad()
def eval_neumf(outputs: str, data_dir: str, split: str, k: int,
               emb_gmf=32, emb_mlp=32, mlp_layers=(256,128,64)):
    users, truths, cands = _load_candidates(outputs, split)
    with open(os.path.join(data_dir, "mappings.pkl"), "rb") as f:
        M = pickle.load(f)
    num_users, num_items = M["num_users"], M["num_items"]

    model = NeuMF(num_users, num_items, emb_gmf=emb_gmf, emb_mlp=emb_mlp, mlp_layers=mlp_layers)
    model.load_state_dict(torch.load(os.path.join(outputs, "neumf.pt"), map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    scores = np.zeros_like(cands, dtype=np.float32)
    for r in tqdm(range(cands.shape[0]), desc=f"Eval NeuMF ({split})"):
        u = int(users[r]); cand = cands[r]
        uu = torch.full((len(cand),), u, dtype=torch.long, device=device)
        ii = torch.tensor(cand, dtype=torch.long, device=device)
        scores[r] = model(uu, ii).detach().cpu().numpy()

    hr, ndcg = hr_ndcg_at_k(scores, k)
    print(f"HR@{k}={hr:.4f}  NDCG@{k}={ndcg:.4f}")
    return hr, ndcg


# ---------- Hybrid NeuMF ----------

@torch.no_grad()
def eval_hybrid(outputs: str, data_dir: str, split: str, k: int,
                emb_gmf=32, emb_mlp=32, feat_proj=32, mlp_layers=(256,128,64)):
    users, truths, cands = _load_candidates(outputs, split)
    with open(os.path.join(data_dir, "mappings.pkl"), "rb") as f:
        M = pickle.load(f)
    num_users, num_items = M["num_users"], M["num_items"]
    item_feats = torch.tensor(np.load(os.path.join(outputs, "item_feats.npy")), dtype=torch.float32)

    model = HybridNeuMF(num_users, num_items, feat_dim=item_feats.shape[1],
                        emb_gmf=emb_gmf, emb_mlp=emb_mlp, feat_proj=feat_proj, mlp_layers=mlp_layers)
    model.load_state_dict(torch.load(os.path.join(outputs, "hybrid_neumf.pt"), map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    item_feats = item_feats.to(device, non_blocking=True)

    scores = np.zeros_like(cands, dtype=np.float32)
    for r in tqdm(range(cands.shape[0]), desc=f"Eval HybridNeuMF ({split})"):
        u = int(users[r]); cand = cands[r]
        uu = torch.full((len(cand),), u, dtype=torch.long, device=device)
        ii = torch.tensor(cand, dtype=torch.long, device=device)
        scores[r] = model(uu, ii, item_feats).detach().cpu().numpy()

    hr, ndcg = hr_ndcg_at_k(scores, k)
    print(f"HR@{k}={hr:.4f}  NDCG@{k}={ndcg:.4f}")
    return hr, ndcg


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["popularity","itemknn","neumf","hybrid"])
    ap.add_argument("--data_dir", default="data_clean")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--k", type=int, default=10)

    # optional arch params (only used if shapes differ)
    ap.add_argument("--emb_gmf", type=int, default=32)
    ap.add_argument("--emb_mlp", type=int, default=32)
    ap.add_argument("--mlp_layers", type=str, default="256,128,64")
    ap.add_argument("--feat_proj", type=int, default=32)

    # itemknn
    ap.add_argument("--use_k_neighbors", type=int, default=50)

    args = ap.parse_args()
    mlp = tuple(int(x) for x in args.mlp_layers.split(",") if x.strip())

    if args.model == "popularity":
        eval_popularity(args.outputs, args.split, args.k)
    elif args.model == "itemknn":
        eval_itemknn(args.outputs, args.data_dir, args.split, args.k, args.use_k_neighbors)
    elif args.model == "neumf":
        eval_neumf(args.outputs, args.data_dir, args.split, args.k, args.emb_gmf, args.emb_mlp, mlp)
    elif args.model == "hybrid":
        eval_hybrid(args.outputs, args.data_dir, args.split, args.k, args.emb_gmf, args.emb_mlp, args.feat_proj, mlp)

if __name__ == "__main__":
    main()
