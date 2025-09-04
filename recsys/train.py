import os, json, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from .data import prepare_all
from .models.neumf import NeuMF
from .models.knn import UserKNN


# ---------- inline evaluator for NeuMF ----------
@torch.no_grad()
def _eval_neumf_current(model: NeuMF, out_dir: str, split: str = "val", k: int = 10) -> dict:
    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))

    device = next(model.parameters()).device
    model.eval()
    scores = np.zeros_like(cands, dtype=float)

    for r in range(cands.shape[0]):
        u = int(users[r]); cand = cands[r]
        uu = torch.full((len(cand),), u, dtype=torch.long, device=device)
        ii = torch.tensor(cand, dtype=torch.long, device=device)
        s = model(uu, ii).detach().cpu().numpy()
        scores[r] = s

    # metrics
    order = np.argsort(-scores, axis=1)
    topk = order[:, :k]
    hits = (topk == 0).any(axis=1).astype(np.float32)
    pos = np.argwhere(order == 0)
    ranks = np.full(order.shape[0], order.shape[1], dtype=np.int64)
    ranks[pos[:, 0]] = pos[:, 1]
    ndcg = np.where(hits > 0, 1.0 / np.log2(ranks + 2), 0.0)

    return {f"HR@{k}": float(hits.mean()), f"NDCG@{k}": float(ndcg.mean())}


# ---------- NeuMF with early stopping ----------
def train_neumf(
    data_dir="data", outputs="outputs",
    epochs=20, batch_size=8192, lr=3e-3, weight_decay=1e-6,
    emb_gmf=16, emb_mlp=32, neg_k=4, seed=42,
    eval_k=10, patience=3, eval_every=1
):
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(outputs, exist_ok=True)

    stats = prepare_all(data_dir=data_dir, out_dir=outputs, neg_k=neg_k, seed=seed)
    pack = np.load(os.path.join(outputs, "train_pairs.npz"))
    u = torch.tensor(pack["u"], dtype=torch.long)
    i = torch.tensor(pack["i"], dtype=torch.long)
    y = torch.tensor(pack["y"], dtype=torch.float32)
    dl = DataLoader(TensorDataset(u, i, y), batch_size=batch_size, shuffle=True, drop_last=False)

    model = NeuMF(stats["num_users"], stats["num_items"], emb_gmf=emb_gmf, emb_mlp=emb_mlp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCELoss()

    best = {"epoch": 0, "metric": 0.0}
    no_improve = 0

    for ep in range(1, epochs + 1):
        # --- train ---
        model.train(); running = 0.0
        for bu, bi, by in tqdm(dl, desc=f"Epoch {ep}/{epochs}"):
            bu, bi, by = bu.to(device), bi.to(device), by.to(device)
            pred = model(bu, bi)
            loss = bce(pred, by)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * len(by)
        print(f"Epoch {ep}: train BCE={running/len(y):.4f}")

        # --- eval ---
        if ep % eval_every == 0:
            metrics = _eval_neumf_current(model, outputs, split="val", k=eval_k)
            ndcg = metrics[f"NDCG@{eval_k}"]; hr = metrics[f"HR@{eval_k}"]
            print(f"  Val HR@{eval_k}={hr:.4f} NDCG@{eval_k}={ndcg:.4f}")
            if ndcg > best["metric"]:
                best.update({"epoch": ep, "metric": ndcg})
                no_improve = 0
                torch.save(model.state_dict(), os.path.join(outputs, "model.pt"))
                with open(os.path.join(outputs, "best_val_metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=2)
                print("  ✅ New best model saved")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"⏹ Early stopping at epoch {ep} (best={best})")
                    break


# ---------- UserKNN baseline ----------
def train_userknn(data_dir="data", outputs="outputs", k=50, seed=42):
    from .data import prepare_all
    from scipy import sparse

    np.random.seed(seed)
    stats = prepare_all(data_dir=data_dir, out_dir=outputs, seed=seed)

    pack = np.load(os.path.join(outputs, "train_pairs.npz"))
    mask = pack["y"] == 1
    u_pos, i_pos = pack["u"][mask], pack["i"][mask]

    model = UserKNN(k=k).fit(u_pos, i_pos, stats["num_users"], stats["num_items"])

    if model.user_sim is None or model.ui is None:
        raise RuntimeError("UserKNN fit failed, got None values")

    os.makedirs(outputs, exist_ok=True)
    np.savez_compressed(os.path.join(outputs, "userknn_model.npz"),
                    user_sim=model.user_sim)
    print("✅ Saved UserKNN model to", outputs)


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", type=str, default="neumf", choices=["neumf", "userknn"])
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--outputs", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)

    # NeuMF args
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--emb_gmf", type=int, default=16)
    ap.add_argument("--emb_mlp", type=int, default=32)
    ap.add_argument("--neg_k", type=int, default=4)
    ap.add_argument("--eval_k", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--eval_every", type=int, default=1)

    # KNN args
    ap.add_argument("--knn_k", type=int, default=50)

    args = ap.parse_args()

    if args.trainer == "neumf":
        train_neumf(
            data_dir=args.data_dir, outputs=args.outputs,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            emb_gmf=args.emb_gmf, emb_mlp=args.emb_mlp,
            neg_k=args.neg_k, seed=args.seed,
            eval_k=args.eval_k, patience=args.patience, eval_every=args.eval_every
        )
    elif args.trainer == "userknn":
        train_userknn(
            data_dir=args.data_dir, outputs=args.outputs,
            k=args.knn_k, seed=args.seed
        )


if __name__ == "__main__":
    main()
