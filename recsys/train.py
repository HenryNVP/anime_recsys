# recsys/train.py
from __future__ import annotations
import os, json, argparse, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from .data import prepare_splits, build_eval_candidates
from .models.popularity import Popularity
from .models.itemknn import ItemKNN
from .models.neumf import NeuMF  # your existing file
from .metrics import hr_ndcg_at_k

def _ensure_candidates(train_df, val_df, test_df, stats, out_dir, seed):
    for split_name, df, z in [("val", val_df, 1), ("test", test_df, 2)]:
        uf = os.path.join(out_dir, f"{split_name}_users.npy")
        cf = os.path.join(out_dir, f"{split_name}_cands.npy")
        if not (os.path.exists(uf) and os.path.exists(cf)):
            users, truths, cands = build_eval_candidates(train_df, df, stats["num_items"], num_negs=99, seed=seed+z)
            np.save(uf, users); np.save(os.path.join(out_dir, f"{split_name}_truths.npy"), truths); np.save(cf, cands)

# -------------------- trainers --------------------
def train_popularity(data_dir="data", outputs="outputs", seed=42):
    os.makedirs(outputs, exist_ok=True)
    train, val, test, stats = prepare_splits(data_dir, outputs, seed=seed)
    _ensure_candidates(train, val, test, stats, outputs, seed)
    model = Popularity().fit(train, stats["num_items"])
    model.save(outputs)
    print("✅ Saved popularity model.")

def train_itemknn(data_dir="data", outputs="outputs", max_neighbors=200, seed=42):
    os.makedirs(outputs, exist_ok=True)
    train, val, test, stats = prepare_splits(data_dir, outputs, seed=seed)
    _ensure_candidates(train, val, test, stats, outputs, seed)
    model = ItemKNN(max_neighbors=max_neighbors).fit(train, stats["num_users"], stats["num_items"])
    model.save(outputs)
    print(f"✅ Saved itemknn model (max_neighbors={max_neighbors}).")

def train_neumf(
    data_dir="data", outputs="outputs",
    epochs=12, batch_size=65536, lr=3e-3, weight_decay=1e-6,
    emb_gmf=64, emb_mlp=64, mlp_layers=(256,128,64),
    neg_k=4, eval_k=10, patience=3, seed=42, num_workers=2, amp=True
):
    os.makedirs(outputs, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed)
    train, val, test, stats = prepare_splits(data_dir, outputs, seed=seed)
    _ensure_candidates(train, val, test, stats, outputs, seed)

    # Build/buffer pointwise pairs (simple, on-the-fly)
    from .data import _user_pos_map
    def build_pointwise_pairs(train_df, num_items, neg_k, seed):
        rng = np.random.default_rng(seed)
        pos = _user_pos_map(train_df)
        all_items = np.arange(num_items, dtype=np.int64)
        users, items, labels = [], [], []
        for _, row in train_df.iterrows():
            u, i = int(row.u), int(row.i)
            users += [u]; items += [i]; labels += [1]
            seen = pos.get(u, set())
            seen_arr = np.fromiter(seen, dtype=np.int64) if seen else np.empty(0, dtype=np.int64)
            pool = np.setdiff1d(all_items, seen_arr, assume_unique=True)
            k = min(neg_k, len(pool))
            negs = rng.choice(pool, size=k, replace=False) if k>0 else rng.choice(all_items, size=neg_k, replace=True)
            for j in negs:
                users += [u]; items += [int(j)]; labels += [0]
        return np.array(users), np.array(items), np.array(labels, dtype=np.float32)

    u_np, i_np, y_np = build_pointwise_pairs(train, stats["num_items"], neg_k=neg_k, seed=seed)

    dl = DataLoader(
        TensorDataset(torch.tensor(u_np, dtype=torch.long),
                      torch.tensor(i_np, dtype=torch.long),
                      torch.tensor(y_np, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    model = NeuMF(stats["num_users"], stats["num_items"], emb_gmf=emb_gmf, emb_mlp=emb_mlp, mlp_layers=mlp_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_ndcg, bad = 0.0, 0
    for ep in range(1, epochs+1):
        model.train(); running = 0.0
        for bu, bi, by in tqdm(dl, desc=f"Epoch {ep}/{epochs}"):
            bu, bi, by = bu.to(device), bi.to(device), by.to(device)
            with torch.cuda.amp.autocast(enabled=amp):
                pred = model(bu, bi)
                loss = bce(pred, by)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            running += loss.item() * len(by)
        print(f"Epoch {ep} BCE={running/len(y_np):.4f}")

        # quick val eval
        hr, ndcg = _eval_neumf_current(model, outputs, split="val", k=eval_k)
        print(f"  Val HR@{eval_k}={hr:.4f} NDCG@{eval_k}={ndcg:.4f}")
        if ndcg > best_ndcg:
            best_ndcg, bad = ndcg, 0
            torch.save(model.state_dict(), os.path.join(outputs, "neumf.pt"))
            json.dump({"epoch": ep, "val_hr": hr, "val_ndcg": ndcg},
                      open(os.path.join(outputs, "neumf_meta.json"), "w"), indent=2)
            print("  ✅ Saved best NeuMF")
        else:
            bad += 1
            if bad >= patience:
                print("⏹ Early stopping")
                break

def _eval_neumf_current(model, out_dir: str, split="val", k=10):
    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))
    device = next(model.parameters()).device
    model.eval()
    scores = np.zeros_like(cands, dtype=float)
    with torch.no_grad():
        for r in range(cands.shape[0]):
            u = int(users[r]); cand = cands[r]
            uu = torch.full((len(cand),), u, dtype=torch.long, device=device)
            ii = torch.tensor(cand, dtype=torch.long, device=device)
            scores[r] = model(uu, ii).detach().cpu().numpy()
    return hr_ndcg_at_k(scores, k=k)

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Train baselines & NeuMF")
    ap.add_argument("--trainer", choices=["popularity", "itemknn", "neumf"], required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    # itemknn
    ap.add_argument("--max_neighbors", type=int, default=200)
    # neumf
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--emb_gmf", type=int, default=64)
    ap.add_argument("--emb_mlp", type=int, default=64)
    ap.add_argument("--mlp_layers", type=str, default="256,128,64")
    ap.add_argument("--neg_k", type=int, default=4)
    ap.add_argument("--eval_k", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()

    if args.trainer == "popularity":
        train_popularity(args.data_dir, args.outputs, seed=args.seed)
    elif args.trainer == "itemknn":
        train_itemknn(args.data_dir, args.outputs, max_neighbors=args.max_neighbors, seed=args.seed)
    elif args.trainer == "neumf":
        layers = tuple(int(x) for x in args.mlp_layers.split(",") if x.strip())
        train_neumf(args.data_dir, args.outputs, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    weight_decay=args.weight_decay, emb_gmf=args.emb_gmf, emb_mlp=args.emb_mlp,
                    mlp_layers=layers, neg_k=args.neg_k, eval_k=args.eval_k, patience=args.patience, seed=args.seed)

if __name__ == "__main__":
    main()
