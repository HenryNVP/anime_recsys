# recsys/train.py
from __future__ import annotations
import os, json, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_matrix

from .data import load_splits, build_eval_candidates, build_pointwise_pairs
from .metrics import hr_ndcg_at_k
from .features import make_item_features
from .models.neumf import NeuMF
from .models.hybrid_neumf import HybridNeuMF


# ---------- shared helpers ----------

def _ensure_candidates(train_df, val_df, test_df, stats, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    for name, df, off in (("val", val_df, 1), ("test", test_df, 2)):
        u_path = os.path.join(out_dir, f"{name}_users.npy")
        c_path = os.path.join(out_dir, f"{name}_cands.npy")
        t_path = os.path.join(out_dir, f"{name}_truths.npy")
        if not (os.path.exists(u_path) and os.path.exists(c_path) and os.path.exists(t_path)):
            users, truths, cands = build_eval_candidates(
                train=train_df, split=df,
                num_items=stats["num_items"], num_negs=99, seed=seed + off
            )
            np.save(u_path, users)
            np.save(t_path, truths)
            np.save(c_path, cands)


# ---------- Popularity trainer ----------

def train_popularity(data_dir="data_clean", outputs="outputs", seed=42):
    os.makedirs(outputs, exist_ok=True)
    train_df, val_df, test_df, stats = load_splits(data_dir)
    _ensure_candidates(train_df, val_df, test_df, stats, outputs, seed)

    # item popularity = count in train
    pop = np.zeros(stats["num_items"], dtype=np.int64)
    # fast bincount over items
    counts = np.bincount(train_df["i"].to_numpy(), minlength=stats["num_items"])
    pop[:len(counts)] = counts
    np.save(os.path.join(outputs, "popularity.npy"), pop)
    print(f"✅ Saved popularity counts to {os.path.join(outputs, 'popularity.npy')}")


# ---------- ItemKNN trainer (cosine) ----------

def _build_csr_ui(train_df, num_users, num_items):
    # rows = users, cols = items, values = 1
    rows = train_df["u"].to_numpy(dtype=np.int64)
    cols = train_df["i"].to_numpy(dtype=np.int64)
    data = np.ones(len(train_df), dtype=np.float32)
    ui = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))
    return ui

def train_itemknn(
    data_dir: str = "data_clean",
    outputs: str = "outputs",
    max_neighbors: int = 200,
    seed: int = 42,
):
    """
    Compute item-item cosine similarity from the train split and keep Top-N
    neighbors per item. Saves:
      - outputs/itemknn_topk.npz   (CSR: data/indices/indptr/shape for Top-N sim)
      - outputs/ui_csr.npz         (CSR: user-item interactions, for scoring)
    """
    os.makedirs(outputs, exist_ok=True)
    train_df, val_df, test_df, stats = load_splits(data_dir)
    _ensure_candidates(train_df, val_df, test_df, stats, outputs, seed)

    num_users, num_items = stats["num_users"], stats["num_items"]

    # --- build CSR user->item matrix (U x I) with 1s for interactions
    rows = train_df["u"].to_numpy(dtype=np.int64)
    cols = train_df["i"].to_numpy(dtype=np.int64)
    data = np.ones(len(train_df), dtype=np.float32)
    ui = csr_matrix((data, (rows, cols)), shape=(num_users, num_items))  # (U, I)

    # --- transpose: item->user matrix (I x U), then L2-normalize rows
    iu = ui.T.tocsr()  # ensure CSR so we can slice rows
    row_norms = np.sqrt(iu.multiply(iu).sum(axis=1)).A1 + 1e-12
    iu_norm = iu.multiply(1.0 / row_norms[:, None]).tocsr()  # keep CSR

    # --- compute Top-N cosine neighbors per item in blocks
    I, N = num_items, int(max_neighbors)
    indptr = [0]
    indices = []
    values = []

    block = max(512, min(2048, I))  # adaptive block size
    pbar = tqdm(range(0, I, block), desc="ItemKNN (cosine, top-N)")
    for start in pbar:
        stop = min(I, start + block)
        # (B x U) @ (U x I) -> (B x I), remains sparse
        block_sim = (iu_norm[start:stop, :] @ iu_norm.T).tocsr()

        # iterate each row in the block and keep top-N (excluding self)
        for r in range(block_sim.shape[0]):
            global_row = start + r
            row = block_sim.getrow(r)
            idx = row.indices
            val = row.data

            # remove self-similarity if present
            mask = idx != global_row
            idx = idx[mask]
            val = val[mask]

            if idx.size == 0:
                indptr.append(indptr[-1])
                continue

            # top-N via partial selection (argpartition), then sort descending
            if idx.size > N:
                top = np.argpartition(-val, N - 1)[:N]
                idx = idx[top]
                val = val[top]

            order = np.argsort(-val)
            idx = idx[order]
            val = val[order]

            indices.extend(idx.tolist())
            values.extend(val.astype(np.float32).tolist())
            indptr.append(len(indices))

    # pack Top-N matrix as CSR (I x I)
    itemknn_indptr = np.asarray(indptr, dtype=np.int64)
    itemknn_indices = np.asarray(indices, dtype=np.int64)
    itemknn_data = np.asarray(values, dtype=np.float32)
    itemknn_shape = np.array((I, I), dtype=np.int64)

    np.savez_compressed(
        os.path.join(outputs, "itemknn_topk.npz"),
        data=itemknn_data,
        indices=itemknn_indices,
        indptr=itemknn_indptr,
        shape=itemknn_shape,
    )

    # snapshot UI for fast scoring later
    np.savez_compressed(
        os.path.join(outputs, "ui_csr.npz"),
        data=ui.data,
        indices=ui.indices,
        indptr=ui.indptr,
        shape=np.array(ui.shape, dtype=np.int64),
    )

    print("✅ Saved ItemKNN to outputs/itemknn_topk.npz and ui_csr.npz")


# ---------- NeuMF trainer ----------

@torch.no_grad()
def _eval_neumf_current(model, out_dir, split="val", k=10):
    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))
    device = next(model.parameters()).device
    model.eval()
    scores = np.zeros_like(cands, dtype=np.float32)
    for r in range(cands.shape[0]):
        u = int(users[r]); cand = cands[r]
        uu = torch.full((len(cand),), u, dtype=torch.long, device=device)
        ii = torch.tensor(cand, dtype=torch.long, device=device)
        scores[r] = model(uu, ii).detach().cpu().numpy()
    return hr_ndcg_at_k(scores, k=k)

def train_neumf(
    data_dir="data_clean", outputs="outputs",
    epochs=8, batch_size=131072, lr=3e-3, weight_decay=1e-6,
    emb_gmf=64, emb_mlp=64, mlp_layers=(256,128,64),
    neg_k=4, eval_k=10, patience=2, seed=42,
    num_workers=2, amp=True
):
    os.makedirs(outputs, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    train_df, val_df, test_df, stats = load_splits(data_dir)
    _ensure_candidates(train_df, val_df, test_df, stats, outputs, seed)

    u, i, y = build_pointwise_pairs(train_df, stats["num_items"], neg_k=neg_k, seed=seed)
    dl = DataLoader(TensorDataset(
        torch.tensor(u, dtype=torch.long),
        torch.tensor(i, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32)
    ), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuMF(stats["num_users"], stats["num_items"],
                  emb_gmf=emb_gmf, emb_mlp=emb_mlp, mlp_layers=mlp_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    best_ndcg, bad = 0.0, 0
    for ep in range(1, epochs+1):
        model.train(); running = 0.0
        for bu, bi, by in tqdm(dl, desc=f"[NeuMF] epoch {ep}/{epochs}"):
            bu, bi, by = bu.to(device, non_blocking=True), bi.to(device, non_blocking=True), by.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=amp):
                pred = model(bu, bi)
                loss = bce(pred, by)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            running += loss.item() * len(by)

        train_bce = running / len(y)
        hr, ndcg = _eval_neumf_current(model, outputs, split="val", k=eval_k)
        print(f"  train_bce={train_bce:.4f}  val HR@{eval_k}={hr:.4f}  NDCG@{eval_k}={ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg, bad = ndcg, 0
            torch.save(model.state_dict(), os.path.join(outputs, "neumf.pt"))
            json.dump({"epoch": ep, "val_hr": hr, "val_ndcg": ndcg},
                      open(os.path.join(outputs, "neumf_meta.json"), "w"), indent=2)
            print("  ✅ Saved best NeuMF")
        else:
            bad += 1
            if bad >= patience:
                print(f"⏹ Early stopping (patience={patience})")
                break


# ---------- Hybrid NeuMF trainer ----------

@torch.no_grad()
def _eval_hybrid_current(model, item_feats, out_dir, split="val", k=10):
    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))
    device = next(model.parameters()).device
    model.eval()
    scores = np.zeros_like(cands, dtype=np.float32)
    for r in range(cands.shape[0]):
        u = int(users[r]); cand = cands[r]
        uu = torch.full((len(cand),), u, dtype=torch.long, device=device)
        ii = torch.tensor(cand, dtype=torch.long, device=device)
        scores[r] = model(uu, ii, item_feats).detach().cpu().numpy()
    return hr_ndcg_at_k(scores, k=k)

def train_hybrid_neumf(
    data_dir="data_clean", outputs="outputs",
    epochs=8, batch_size=131072, lr=3e-3, weight_decay=1e-6,
    emb_gmf=64, emb_mlp=64, feat_proj=32, mlp_layers=(256,128,64),
    neg_k=4, eval_k=10, patience=2, seed=42,
    num_workers=2, amp=True, dropout=0.0
):
    os.makedirs(outputs, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)

    train_df, val_df, test_df, stats = load_splits(data_dir)
    _ensure_candidates(train_df, val_df, test_df, stats, outputs, seed)

    # features
    if not os.path.exists(os.path.join(outputs, "item_feats.npy")):
        make_item_features(data_dir, outputs)
    item_feats = torch.tensor(np.load(os.path.join(outputs, "item_feats.npy")), dtype=torch.float32)

    u, i, y = build_pointwise_pairs(train_df, stats["num_items"], neg_k=neg_k, seed=seed)
    dl = DataLoader(TensorDataset(
        torch.tensor(u, dtype=torch.long),
        torch.tensor(i, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32)
    ), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridNeuMF(stats["num_users"], stats["num_items"], feat_dim=item_feats.shape[1],
                        emb_gmf=emb_gmf, emb_mlp=emb_mlp, feat_proj=feat_proj,
                        mlp_layers=mlp_layers, dropout=dropout).to(device)
    item_feats = item_feats.to(device, non_blocking=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    best_ndcg, bad = 0.0, 0
    for ep in range(1, epochs+1):
        model.train(); running = 0.0
        for bu, bi, by in tqdm(dl, desc=f"[HybridNeuMF] epoch {ep}/{epochs}"):
            bu, bi, by = bu.to(device, non_blocking=True), bi.to(device, non_blocking=True), by.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=amp):
                pred = model(bu, bi, item_feats)
                loss = bce(pred, by)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            running += loss.item() * len(by)

        train_bce = running / len(y)
        hr, ndcg = _eval_hybrid_current(model, item_feats, outputs, split="val", k=eval_k)
        print(f"  train_bce={train_bce:.4f}  val HR@{eval_k}={hr:.4f}  NDCG@{eval_k}={ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg, bad = ndcg, 0
            torch.save(model.state_dict(), os.path.join(outputs, "hybrid_neumf.pt"))
            json.dump({"epoch": ep, "val_hr": hr, "val_ndcg": ndcg,
              "config": {"emb_gmf": emb_gmf, "emb_mlp": emb_mlp, "mlp_layers": list(mlp_layers)}},
              open(os.path.join(outputs, "neumf_meta.json"), "w"), indent=2)
            print("  ✅ Saved best HybridNeuMF")
        else:
            bad += 1
            if bad >= patience:
                print(f"⏹ Early stopping (patience={patience})")
                break


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", required=True, choices=["popularity","itemknn","neumf","hybrid"])
    ap.add_argument("--data_dir", default="data_clean")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--seed", type=int, default=42)

    # itemknn
    ap.add_argument("--max_neighbors", type=int, default=200)

    # shared NN params
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=131072)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--emb_gmf", type=int, default=64)
    ap.add_argument("--emb_mlp", type=int, default=64)
    ap.add_argument("--mlp_layers", type=str, default="256,128,64")
    ap.add_argument("--neg_k", type=int, default=4)
    ap.add_argument("--k_eval", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--feat_proj", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)

    args = ap.parse_args()
    layers = tuple(int(x) for x in args.mlp_layers.split(",") if x.strip())

    if args.trainer == "popularity":
        train_popularity(args.data_dir, args.outputs, args.seed)
    elif args.trainer == "itemknn":
        train_itemknn(args.data_dir, args.outputs, args.max_neighbors, args.seed)
    elif args.trainer == "neumf":
        train_neumf(args.data_dir, args.outputs, args.epochs, args.batch_size, args.lr, args.weight_decay,
                    args.emb_gmf, args.emb_mlp, layers, args.neg_k, args.k_eval, args.patience, args.seed,
                    args.num_workers, args.amp)
    elif args.trainer == "hybrid":
        train_hybrid_neumf(args.data_dir, args.outputs, args.epochs, args.batch_size, args.lr, args.weight_decay,
                           args.emb_gmf, args.emb_mlp, args.feat_proj, layers, args.neg_k, args.k_eval,
                           args.patience, args.seed, args.num_workers, args.amp, args.dropout)

if __name__ == "__main__":
    main()
