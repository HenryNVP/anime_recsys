# recsys/eval.py
from __future__ import annotations
import os, argparse, pickle, json
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
    users = np.load(os.path.join(outputs, f"{split}_users.npy"))
    truths = np.load(os.path.join(outputs, f"{split}_truths.npy"))
    cands = np.load(os.path.join(outputs, f"{split}_cands.npy"))

    # 1) try to read sizes from meta files
    cfg = None
    def _try(path):
        try:
            if os.path.exists(path):
                return json.load(open(path))
        except Exception:
            pass
        return None

    # direct meta (from plain train.py)
    meta = _try(os.path.join(outputs, "neumf_meta.json"))
    # tuner’s “selected” pointer
    sel  = _try(os.path.join(outputs, "neumf_selected_meta.json"))
    if sel and "selected_from" in sel:
        best_meta = _try(os.path.join(sel["selected_from"], "best_meta.json"))
        if best_meta and "config" in best_meta:
            cfg = best_meta["config"]
    if (not cfg) and meta:
        # plain meta may not have config, but try anyway
        if "config" in meta:
            cfg = meta["config"]

    # 2) default to CLI args; override if cfg present
    if cfg:
        try:
            emb_gmf = int(cfg.get("emb_gmf", emb_gmf))
            emb_mlp = int(cfg.get("emb_mlp", emb_mlp))
            # mlp_layers may be list or "256,128,64" or "256-128-64"
            raw = cfg.get("mlp_layers", mlp_layers)
            if isinstance(raw, (list, tuple)):
                mlp_layers = tuple(int(x) for x in raw)
            else:
                mlp_layers = tuple(int(x) for x in str(raw).replace("-", ",").split(",") if str(x).strip())
        except Exception:
            pass

    # 3) if still wrong, infer from checkpoint shapes
    ckpt_path = os.path.join(outputs, "neumf.pt")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Try to infer emb dims from embedding weights
    try:
        # Embedding weight shapes: (num_embeddings, embedding_dim)
        gmf_user_w = state["user_gmf.weight"]  # [num_users, emb_gmf]
        gmf_item_w = state["item_gmf.weight"]
        mlp_user_w = state["user_mlp.weight"]  # [num_users, emb_mlp]
        mlp_item_w = state["item_mlp.weight"]
        emb_gmf = gmf_user_w.shape[1]
        emb_mlp = mlp_user_w.shape[1]
        # Infer MLP sizes from first linear layer
        # First MLP layer weight: [hidden1, 2*emb_mlp]
        # Subsequent layers give hidden sequence
        mlp_keys = sorted([k for k in state.keys() if k.startswith("mlp.") and k.endswith(".weight")],
                          key=lambda s: int(s.split(".")[1]))
        inferred = []
        for idx, kname in enumerate(mlp_keys):
            w = state[kname]
            out_dim, in_dim = w.shape
            if idx == 0:
                # sanity: in_dim should be 2*emb_mlp (ignore if mismatch)
                inferred.append(out_dim)
            else:
                inferred.append(out_dim)
        if inferred:
            mlp_layers = tuple(int(x) for x in inferred)
    except Exception:
        pass

    # Build model with inferred/final sizes
    with open(os.path.join(data_dir, "mappings.pkl"), "rb") as f:
        M = pickle.load(f)
    num_users, num_items = M["num_users"], M["num_items"]

    model = NeuMF(num_users, num_items, emb_gmf=emb_gmf, emb_mlp=emb_mlp, mlp_layers=mlp_layers)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
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
                emb_gmf=64, emb_mlp=64, feat_proj=32, mlp_layers=(256,128,64)):
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
    ap.add_argument("--emb_gmf", type=int, default=64)
    ap.add_argument("--emb_mlp", type=int, default=64)
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
