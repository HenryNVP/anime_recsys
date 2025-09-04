# recsys/train.py
from __future__ import annotations
import os, json, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# ---- our utils (single source of truth for splits/candidates) ----
from .data import (
    prepare_splits,            # returns train_df, val_df, test_df, stats
    build_pointwise_pairs,     # for NeuMF's BCE training
    build_eval_candidates,     # create [true + negatives] candidate lists
)

# ---- models ----
from .models.neumf import NeuMF
from .models.knn import UserKNN


# ===================== Small inline evaluator for NeuMF =====================
@torch.no_grad()
def _eval_neumf_current(model: NeuMF, out_dir: str, split: str = "val", k: int = 10) -> dict:
    """
    Evaluate the *in-memory* model on cached candidate arrays.
    Assumes {split}_users.npy and {split}_cands.npy exist in out_dir.
    """
    import numpy as np
    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))

    device = next(model.parameters()).device
    model.eval()
    scores = np.zeros_like(cands, dtype=float)

    for r in range(cands.shape[0]):
        u = int(users[r]); cand = cands[r]
        uu = torch.full((len(cand),), u, dtype=torch.long, device=device)
        ii = torch.tensor(cand, dtype=torch.long, device=device)
        scores[r] = model(uu, ii).detach().cpu().numpy()

    # ranking metrics (truth is always at index 0 in each row of cands)
    order = np.argsort(-scores, axis=1)       # desc
    topk = order[:, :k]
    hits = (topk == 0).any(axis=1).astype(np.float32)
    pos = np.argwhere(order == 0)             # where index 0 (truth) appears
    ranks = np.full(order.shape[0], order.shape[1], dtype=np.int64)
    ranks[pos[:, 0]] = pos[:, 1]
    ndcg = np.where(ranks < k, 1.0 / np.log2(ranks + 2), 0.0)

    return {f"HR@{k}": float(hits.mean()), f"NDCG@{k}": float(ndcg.mean())}


# ============================== NeuMF trainer ===============================
def _ensure_eval_candidates(train_df, val_df, test_df, stats, out_dir: str, seed: int):
    """Create and cache candidate arrays once (shared by all models)."""
    for split_name, df in (("val", val_df), ("test", test_df)):
        users_f = os.path.join(out_dir, f"{split_name}_users.npy")
        cands_f = os.path.join(out_dir, f"{split_name}_cands.npy")
        if not (os.path.exists(users_f) and os.path.exists(cands_f)):
            users, truths, cands = build_eval_candidates(
                train=train_df,
                split=df,
                num_items=stats["num_items"],
                num_negs=99,
                seed=seed + (1 if split_name == "val" else 2),
            )
            np.save(users_f, users)
            np.save(os.path.join(out_dir, f"{split_name}_truths.npy"), truths)
            np.save(cands_f, cands)

def train_neumf(
    data_dir="data",
    outputs="outputs",
    epochs=10,
    batch_size=65536,
    lr=3e-3,
    weight_decay=1e-6,
    emb_gmf=64,
    emb_mlp=64,
    mlp_layers=(256, 128, 64),
    neg_k=4,
    seed=42,
    eval_k=10,
    patience=3,
    eval_every=1,
    amp=True,
    grad_accum=1,
    num_workers=2,
):
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(outputs, exist_ok=True)

    # ---- unified splits ----
    train_df, val_df, test_df, stats = prepare_splits(
        data_dir=data_dir, out_dir=outputs, seed=seed
    )
    _ensure_eval_candidates(train_df, val_df, test_df, stats, outputs, seed)

    # ---- cache or build pointwise (u,i,y) for BCE ----
    pairs_path = os.path.join(outputs, "train_pairs.npz")
    if os.path.exists(pairs_path):
        pack = np.load(pairs_path)
        u_np, i_np, y_np = pack["u"], pack["i"], pack["y"]
    else:
        u_np, i_np, y_np = build_pointwise_pairs(
            train=train_df, num_items=stats["num_items"], neg_k=neg_k, seed=seed
        )
        np.savez_compressed(pairs_path, u=u_np, i=i_np, y=y_np)

    u = torch.tensor(u_np, dtype=torch.long)
    i = torch.tensor(i_np, dtype=torch.long)
    y = torch.tensor(y_np, dtype=torch.float32)

    dl = DataLoader(
        TensorDataset(u, i, y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    torch.backends.cudnn.benchmark = True

    # ---- model/opt/loss ----
    model = NeuMF(
        stats["num_users"], stats["num_items"],
        emb_gmf=emb_gmf, emb_mlp=emb_mlp, mlp_layers=tuple(mlp_layers)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_ndcg, bad = 0.0, 0

    for ep in range(1, epochs + 1):
        model.train(); running = 0.0
        opt.zero_grad(set_to_none=True)

        for step, (bu, bi, by) in enumerate(tqdm(dl, desc=f"Epoch {ep}/{epochs}")):
            bu = bu.to(device, non_blocking=True)
            bi = bi.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp):
                pred = model(bu, bi)
                loss = bce(pred, by) / grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

            running += loss.item() * len(by) * grad_accum  # undo division for logging

        train_bce = running / len(y)
        print(f"Epoch {ep}: train BCE={train_bce:.4f}")

        # ---- eval + early stopping ----
        if ep % eval_every == 0:
            metrics = _eval_neumf_current(model, outputs, split="val", k=eval_k)
            hr, ndcg = metrics[f"HR@{eval_k}"], metrics[f"NDCG@{eval_k}"]
            print(f"  Val HR@{eval_k}={hr:.4f}  NDCG@{eval_k}={ndcg:.4f}")

            if ndcg > best_ndcg:
                best_ndcg, bad = ndcg, 0
                torch.save(model.state_dict(), os.path.join(outputs, "model.pt"))
                with open(os.path.join(outputs, "best_val_metrics.json"), "w") as f:
                    json.dump({"epoch": ep, **metrics}, f, indent=2)
                print("  ✅ New best model saved")
            else:
                bad += 1
                print(f"  (no improvement) patience {bad}/{patience}")
                if bad >= patience:
                    print(f"⏹ Early stopping at epoch {ep} (best NDCG={best_ndcg:.4f})")
                    break

    # ensure a model file exists even if no improvement recorded
    if not os.path.exists(os.path.join(outputs, "model.pt")):
        torch.save(model.state_dict(), os.path.join(outputs, "model.pt"))


# ============================== UserKNN trainer =============================
def train_userknn(
    data_dir="data",
    outputs="outputs",
    k_neighbors=50,
    seed=42,
):
    """
    Memory-based CF. Uses only TRAIN positives.
    Skips negative sampling entirely.
    Saves UI CSR + per-user norms (compact), and writes meta with k.
    """
    os.makedirs(outputs, exist_ok=True)
    np.random.seed(seed)

    # unified splits (same as NeuMF)
    train_df, val_df, test_df, stats = prepare_splits(
        data_dir=data_dir, out_dir=outputs, seed=seed
    )
    # also ensure candidate arrays exist (used in eval)
    _ensure_eval_candidates(train_df, val_df, test_df, stats, outputs, seed)

    # positives from train
    u_pos = train_df["u"].to_numpy()
    i_pos = train_df["i"].to_numpy()

    model = UserKNN(k_neighbors==k_neighbors).fit(
        u_pos, i_pos, stats["num_users"], stats["num_items"]
    )

    # Save compact CSR + norms
    ui = model.ui
    if ui is None or model.user_norms is None:
        raise ValueError("UserKNN.ui is None; call fit() first")
    else:
        n_users, n_items = ui.shape

    np.savez_compressed(
        os.path.join(outputs, "userknn_ui.npz"),
        ui_data=np.asarray(ui.data),
        ui_indices=np.asarray(ui.indices),
        ui_indptr=np.asarray(ui.indptr),
        ui_n_users=np.asarray([n_users], dtype=np.int64),
        ui_n_items=np.asarray([n_items], dtype=np.int64),
        ui_norms=np.asarray(model.user_norms),
    )
    with open(os.path.join(outputs, "userknn_meta.json"), "w") as f:
        json.dump({"k_neighbors": int(k_neighbors)}, f)
    print(f"✅ UserKNN saved (k={k_neighbors}) -> outputs/userknn_ui.npz")


# ================================== CLI ====================================
def main():
    ap = argparse.ArgumentParser(description="Train recommenders on Anime dataset")
    ap.add_argument("--trainer", type=str, default="neumf", choices=["neumf", "userknn"])
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--outputs", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)

    # NeuMF args
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--emb_gmf", type=int, default=64)
    ap.add_argument("--emb_mlp", type=int, default=64)
    ap.add_argument("--mlp_layers", type=str, default="256,128,64")
    ap.add_argument("--neg_k", type=int, default=4)
    ap.add_argument("--eval_k", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)

    # KNN args
    ap.add_argument("--knn_k", type=int, default=50)

    args = ap.parse_args()

    if args.trainer == "neumf":
        layers = tuple(int(x) for x in args.mlp_layers.split(",") if x.strip())
        train_neumf(
            data_dir=args.data_dir,
            outputs=args.outputs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            emb_gmf=args.emb_gmf,
            emb_mlp=args.emb_mlp,
            mlp_layers=layers,
            neg_k=args.neg_k,
            seed=args.seed,
            eval_k=args.eval_k,
            patience=args.patience,
            eval_every=args.eval_every,
            amp=args.amp,
            grad_accum=args.grad_accum,
            num_workers=args.num_workers,
        )
    elif args.trainer == "userknn":
        train_userknn(
            data_dir=args.data_dir,
            outputs=args.outputs,
            k_neighbors=args.knn_k,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
