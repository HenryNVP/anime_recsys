# recsys/eval.py (patches)
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


def _maybe_cfg_from_meta(outputs: str, names: list[str]) -> dict | None:
    for name in names:
        path = os.path.join(outputs, name)
        if os.path.exists(path):
            try:
                cfg = json.load(open(path))
                if "config" in cfg:   # tuner/ trainer may nest it
                    return cfg["config"]
                return cfg
            except Exception:
                pass
    return None

def _infer_neumf_shapes_from_ckpt(state: dict) -> tuple[int,int,tuple[int,...]]:
    # state may be the raw state_dict or a { 'state_dict': ... }
    if "state_dict" in state: state = state["state_dict"]
    emb_gmf = state["user_gmf.weight"].shape[1]
    emb_mlp = state["user_mlp.weight"].shape[1]
    mlp_w = [k for k in state.keys() if k.startswith("mlp.") and k.endswith(".weight")]
    mlp_w.sort(key=lambda s: int(s.split(".")[1]))  # mlp.0.weight, mlp.1.weight, ...
    inferred = []
    for k in mlp_w:
        w = state[k]
        inferred.append(int(w.shape[0]))  # out_dim of each layer
    mlp_layers = tuple(inferred) if inferred else (256,128,64)
    return emb_gmf, emb_mlp, mlp_layers

def _infer_hybrid_shapes_from_ckpt(state: dict) -> tuple[int,int,int,tuple[int,...]]:
    if "state_dict" in state: state = state["state_dict"]
    emb_gmf = state["user_gmf.weight"].shape[1]
    emb_mlp = state["user_mlp.weight"].shape[1]
    # feature projection: first linear on item feature tower
    # expected key names may differ; adjust if your module names differ
    feat_proj = None
    for k in state.keys():
        if "feat_proj" in k and k.endswith(".weight"):
            feat_proj = int(state[k].shape[0])
            break
    if feat_proj is None:
        feat_proj = 32

    mlp_w = [k for k in state.keys() if k.startswith("mlp.") and k.endswith(".weight")]
    mlp_w.sort(key=lambda s: int(s.split(".")[1]))
    inferred = [int(state[k].shape[0]) for k in mlp_w] if mlp_w else [256,128,64]
    return emb_gmf, emb_mlp, feat_proj, tuple(inferred)

# ---------------- NeuMF ----------------

@torch.no_grad()
def eval_neumf(outputs: str, data_dir: str, split: str, k: int,
               emb_gmf=32, emb_mlp=32, mlp_layers=(256,128,64)):
    users, truths, cands = _load_candidates(outputs, split)

    # 1) cfg from meta (trainer or tuner)
    cfg = None
    meta_candidates = [
        "neumf_selected_meta.json",   # from tuner copy-up
        "neumf_meta.json",            # from simple trainer
    ]
    # If selected_meta points to a run dir, read its best_meta.json
    sel = _maybe_cfg_from_meta(outputs, ["neumf_selected_meta.json"])
    if sel and "selected_from" in sel:
        best_meta = os.path.join(sel["selected_from"], "best_meta.json")
        if os.path.exists(best_meta):
            j = json.load(open(best_meta))
            cfg = j.get("config")

    if not cfg:
        cfg = _maybe_cfg_from_meta(outputs, meta_candidates)

    if cfg:
        emb_gmf = int(cfg.get("emb_gmf", emb_gmf))
        emb_mlp = int(cfg.get("emb_mlp", emb_mlp))
        raw = cfg.get("mlp_layers", mlp_layers)
        if isinstance(raw, (list, tuple)):
            mlp_layers = tuple(int(x) for x in raw)
        else:
            mlp_layers = tuple(int(x) for x in str(raw).replace("-",",").split(",") if x.strip())

    # 2) if still wrong, infer from checkpoint
    ckpt_path = os.path.join(outputs, "neumf.pt")
    state = torch.load(ckpt_path, map_location="cpu")
    try:
        e_g, e_m, ml = _infer_neumf_shapes_from_ckpt(state)
        emb_gmf, emb_mlp, mlp_layers = e_g, e_m, ml
    except Exception:
        pass

    with open(os.path.join(data_dir, "mappings.pkl"), "rb") as f:
        M = pickle.load(f)
    model = NeuMF(M["num_users"], M["num_items"], emb_gmf=emb_gmf, emb_mlp=emb_mlp, mlp_layers=mlp_layers)
    model.load_state_dict(state if isinstance(state, dict) and "state_dict" not in state else state["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    scores = np.zeros_like(cands, dtype=np.float32)
    for r in tqdm(range(cands.shape[0]), desc=f"Eval NeuMF ({split})"):
        uu = torch.full((cands.shape[1],), int(users[r]), dtype=torch.long, device=device)
        ii = torch.tensor(cands[r], dtype=torch.long, device=device)
        scores[r] = model(uu, ii).detach().cpu().numpy()

    hr, ndcg = hr_ndcg_at_k(scores, k)
    print(f"HR@{k}={hr:.4f}  NDCG@{k}={ndcg:.4f}")
    return hr, ndcg

# ---------------- Hybrid ----------------

@torch.no_grad()
def eval_hybrid(outputs: str, data_dir: str, split: str, k: int,
                emb_gmf=32, emb_mlp=32, feat_proj=32, mlp_layers=(256,128,64)):
    users, truths, cands = _load_candidates(outputs, split)

    # cfg from meta
    cfg = None
    sel = _maybe_cfg_from_meta(outputs, ["hybrid_selected_meta.json"])
    if sel and "selected_from" in sel:
        best_meta = os.path.join(sel["selected_from"], "best_meta.json")
        if os.path.exists(best_meta):
            j = json.load(open(best_meta))
            cfg = j.get("config")
    if not cfg:
        cfg = _maybe_cfg_from_meta(outputs, ["hybrid_meta.json"])

    if cfg:
        emb_gmf = int(cfg.get("emb_gmf", emb_gmf))
        emb_mlp = int(cfg.get("emb_mlp", emb_mlp))
        feat_proj = int(cfg.get("feat_proj", feat_proj))
        raw = cfg.get("mlp_layers", mlp_layers)
        if isinstance(raw, (list, tuple)):
            mlp_layers = tuple(int(x) for x in raw)
        else:
            mlp_layers = tuple(int(x) for x in str(raw).replace("-",",").split(",") if x.strip())

    # infer from checkpoint if needed
    ckpt_path = os.path.join(outputs, "hybrid_neumf.pt")
    state = torch.load(ckpt_path, map_location="cpu")
    try:
        e_g, e_m, f_p, ml = _infer_hybrid_shapes_from_ckpt(state)
        emb_gmf, emb_mlp, feat_proj, mlp_layers = e_g, e_m, f_p, ml
    except Exception:
        pass

    with open(os.path.join(data_dir, "mappings.pkl"), "rb") as f:
        M = pickle.load(f)
    item_feats = torch.tensor(np.load(os.path.join(outputs, "item_feats.npy")), dtype=torch.float32)

    model = HybridNeuMF(M["num_users"], M["num_items"], feat_dim=item_feats.shape[1],
                        emb_gmf=emb_gmf, emb_mlp=emb_mlp, feat_proj=feat_proj, mlp_layers=mlp_layers)
    sd = state if isinstance(state, dict) and "state_dict" not in state else state["state_dict"]
    model.load_state_dict(sd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    item_feats = item_feats.to(device, non_blocking=True)

    scores = np.zeros_like(cands, dtype=np.float32)
    for r in tqdm(range(cands.shape[0]), desc=f"Eval HybridNeuMF ({split})"):
        uu = torch.full((cands.shape[1],), int(users[r]), dtype=torch.long, device=device)
        ii = torch.tensor(cands[r], dtype=torch.long, device=device)
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

