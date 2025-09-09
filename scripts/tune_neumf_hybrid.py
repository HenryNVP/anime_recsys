from __future__ import annotations
import os, json, argparse, shutil
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from recsys.data import load_splits, build_eval_candidates
from recsys.metrics import hr_ndcg_at_k
from recsys.models.neumf import NeuMF

@dataclass(frozen=True)
class NeuMFConfig:
    emb_gmf: int
    emb_mlp: int
    mlp_layers: Tuple[int, ...]
    lr: float
    neg_k: int
    def slug(self) -> str:
        return f"gmf{self.emb_gmf}_mlp{self.emb_mlp}_layers{'-'.join(map(str,self.mlp_layers))}_lr{self.lr}_neg{self.neg_k}"

# --------- helpers ---------
def ensure_candidates(train_df, val_df, test_df, stats: Dict, out_dir: str, seed: int):
    os.makedirs(out_dir, exist_ok=True)
    for split_name, df, z in (("val", val_df, 1), ("test", test_df, 2)):
        u = os.path.join(out_dir, f"{split_name}_users.npy")
        t = os.path.join(out_dir, f"{split_name}_truths.npy")
        c = os.path.join(out_dir, f"{split_name}_cands.npy")
        if not (os.path.exists(u) and os.path.exists(t) and os.path.exists(c)):
            users, truths, cands = build_eval_candidates(
                train=train_df, split=df,
                num_items=stats["num_items"], num_negs=99, seed=seed+z
            )
            np.save(u, users); np.save(t, truths); np.save(c, cands)

def _pairs_cache_path(outputs: str, neg_k: int) -> str:
    return os.path.join(outputs, f"neumf_pairs_neg{neg_k}.npz")

def build_pointwise_pairs_cached(train_df, num_items: int, neg_k: int, seed: int, outputs: str):
    path = _pairs_cache_path(outputs, neg_k)
    if os.path.exists(path):
        pack = np.load(path)
        u = torch.tensor(pack["u"], dtype=torch.long)
        i = torch.tensor(pack["i"], dtype=torch.long)
        y = torch.tensor(pack["y"], dtype=torch.float32)
        return u, i, y

    rng = np.random.default_rng(seed)
    pos_map = train_df.groupby("u")["i"].apply(set).to_dict()
    all_items = np.arange(num_items, dtype=np.int64)

    users, items, labels = [], [], []
    ui = train_df[["u","i"]].to_numpy(dtype=np.int64)
    for u, i in tqdm(ui, total=ui.shape[0], desc=f"Build pairs (neg_k={neg_k})"):
        users.append(u); items.append(i); labels.append(1.0)
        seen = pos_map.get(int(u), set())
        seen_arr = np.fromiter(seen, dtype=np.int64) if seen else np.empty(0, dtype=np.int64)
        pool = np.setdiff1d(all_items, seen_arr, assume_unique=True)
        k = min(neg_k, len(pool))
        negs = rng.choice(pool, size=k, replace=False) if k>0 else rng.choice(all_items, size=neg_k, replace=True)
        for j in negs:
            users.append(u); items.append(int(j)); labels.append(0.0)

    u_arr = np.asarray(users, dtype=np.int64)
    i_arr = np.asarray(items, dtype=np.int64)
    y_arr = np.asarray(labels, dtype=np.float32)
    np.savez_compressed(path, u=u_arr, i=i_arr, y=y_arr)

    return (torch.tensor(u_arr, dtype=torch.long),
            torch.tensor(i_arr, dtype=torch.long),
            torch.tensor(y_arr, dtype=torch.float32))

@torch.no_grad()
def eval_neumf_current(model: NeuMF, out_dir: str, split: str, k_eval: int) -> Tuple[float,float]:
    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))
    device = next(model.parameters()).device
    model.eval()
    scores = np.zeros_like(cands, dtype=np.float32)
    for r in range(cands.shape[0]):
        u = int(users[r]); cand = cands[r]
        uu = torch.full((len(cand),), u, dtype=torch.long, device=device)
        ii = torch.tensor(cand, dtype=torch.long, device=device)
        scores[r] = model(uu, ii).detach().cpu().numpy()  # logits are fine for ranking
    return hr_ndcg_at_k(scores, k=k_eval)

def train_one_config(
    cfg: NeuMFConfig,
    data_dir: str, outputs: str,
    seed: int, epochs: int, batch_size: int, weight_decay: float,
    k_eval: int, patience: int, num_workers: int, amp: bool,
    resume: bool
):
    # setup dirs
    tune_root = os.path.join(outputs, "neumf_tuning"); os.makedirs(tune_root, exist_ok=True)
    run_dir = os.path.join(tune_root, cfg.slug()); os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "last_ckpt.pt")
    best_meta_path = os.path.join(run_dir, "best_meta.json")

    # data & candidates
    torch.manual_seed(seed); np.random.seed(seed)
    train_df, val_df, test_df, stats = load_splits(data_dir)
    ensure_candidates(train_df, val_df, test_df, stats, outputs, seed)

    # pairs
    u, i, y = build_pointwise_pairs_cached(train_df, stats["num_items"], cfg.neg_k, seed, outputs)
    dl = DataLoader(TensorDataset(u,i,y), batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True, drop_last=False)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuMF(stats["num_users"], stats["num_items"],
                  emb_gmf=cfg.emb_gmf, emb_mlp=cfg.emb_mlp, mlp_layers=cfg.mlp_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    best = {"epoch": 0, "val_hr": 0.0, "val_ndcg": 0.0}
    history = []
    start_epoch = 1

    # resume (optional)
    if resume and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            if amp and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
            best = ckpt.get("best", best)
            history = ckpt.get("history", history)
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"🔁 Resuming {cfg.slug()} from epoch {start_epoch}")
        except Exception as e:
            print("Resume failed, starting fresh:", e)

    # train
    bad = 0
    for ep in range(start_epoch, epochs+1):
        model.train(); running = 0.0
        for bu, bi, by in tqdm(dl, desc=f"[{cfg.slug()}] epoch {ep}/{epochs}"):
            bu, bi, by = bu.to(device, non_blocking=True), bi.to(device, non_blocking=True), by.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=amp):
                logits = model(bu, bi)
                loss = bce(logits, by)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            running += loss.item() * len(by)

        train_bce = running / len(y)
        val_hr, val_ndcg = eval_neumf_current(model, outputs, split="val", k_eval=k_eval)
        history.append({"epoch": ep, "train_bce": float(train_bce),
                        f"val_hr@{k_eval}": float(val_hr), f"val_ndcg@{k_eval}": float(val_ndcg)})

        # checkpoint (for resume)
        torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict() if amp else None,
                    "best": best, "history": history}, ckpt_path)

        # early stop on val NDCG
        if val_ndcg > best["val_ndcg"]:
            best = {"epoch": ep, "val_hr": float(val_hr), "val_ndcg": float(val_ndcg)}
            torch.save(model.state_dict(), os.path.join(run_dir, "neumf.pt"))
            json.dump({"config": asdict(cfg), "best": best}, open(best_meta_path,"w"), indent=2)
            print("  ✅ improved; saved best")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"⏹ early stopping (patience={patience})")
                break

    return {"config": cfg, "best": best, "run_dir": run_dir}

# --------- main ---------
def main():
    ap = argparse.ArgumentParser("Tune NeuMF with early stopping; simple skip/resume")
    ap.add_argument("--data_dir", default="data_clean")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=131072)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--k_eval", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--skip_done", action="store_true", default=True)
    ap.add_argument("--resume", action="store_true", default=True)

    # moderate defaults (expand if you have time/GPU)
    ap.add_argument("--emb_gmf_grid", type=str, default="32,64,128")
    ap.add_argument("--emb_mlp_grid", type=str, default="32,64,128")
    ap.add_argument("--mlp_grid", type=str, default="256-128-64,512-256-128")
    ap.add_argument("--lr_grid", type=str, default="0.003,0.001,0.0005")
    ap.add_argument("--negk_grid", type=str, default="2,4,6")
    ap.add_argument("--summary_csv", default="neumf_tuning_summary.csv")
    args = ap.parse_args()

    emb_gmf_grid = [int(x) for x in args.emb_gmf_grid.split(",") if x.strip()]
    emb_mlp_grid = [int(x) for x in args.emb_mlp_grid.split(",") if x.strip()]
    mlp_grid = [tuple(int(y) for y in s.split("-") if y.strip()) for s in args.mlp_grid.split(",") if s.strip()]
    lr_grid = [float(x) for x in args.lr_grid.split(",") if x.strip()]
    negk_grid = [int(x) for x in args.negk_grid.split(",") if x.strip()]

    configs: List[NeuMFConfig] = [
        NeuMFConfig(eg, em, ml, lr, nk)
        for eg in emb_gmf_grid
        for em in emb_mlp_grid
        for ml in mlp_grid
        for lr in lr_grid
        for nk in negk_grid
    ]
    print(f"Total configs: {len(configs)}")

    rows = []
    for cfg in configs:
        run_dir = os.path.join(args.outputs, "neumf_tuning", cfg.slug())
        best_meta = os.path.join(run_dir, "best_meta.json")
        if args.skip_done and os.path.exists(best_meta):
            print(f"⏭️  Skipping {cfg.slug()} (already finished)")
            try:
                bm = json.load(open(best_meta))
                rows.append({
                    "slug": cfg.slug(), **asdict(cfg),
                    "best_epoch": bm["best"]["epoch"],
                    f"best_val_hr@{args.k_eval}": bm["best"]["val_hr"],
                    f"best_val_ndcg@{args.k_eval}": bm["best"]["val_ndcg"],
                    "run_dir": run_dir,
                })
            except Exception:
                pass
            continue

        res = train_one_config(
            cfg, data_dir=args.data_dir, outputs=args.outputs,
            seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
            weight_decay=args.weight_decay, k_eval=args.k_eval,
            patience=args.patience, num_workers=args.num_workers,
            amp=args.amp, resume=args.resume
        )
        rows.append({
            "slug": cfg.slug(), **asdict(cfg),
            "best_epoch": res["best"]["epoch"],
            f"best_val_hr@{args.k_eval}": res["best"]["val_hr"],
            f"best_val_ndcg@{args.k_eval}": res["best"]["val_ndcg"],
            "run_dir": res["run_dir"],
        })

    df = pd.DataFrame(rows).sort_values(by=f"best_val_ndcg@{args.k_eval}", ascending=False)
    out_csv = os.path.join(args.outputs, args.summary_csv)
    df.to_csv(out_csv, index=False)
    print("\n=== NeuMF Tuning Summary (sorted by val NDCG) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n✅ Saved summary CSV to {out_csv}")

    # copy best checkpoint up one level
    if len(df):
        best_dir = df.iloc[0]["run_dir"]
        src = os.path.join(best_dir, "neumf.pt")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.outputs, "neumf.pt"))
            json.dump({"selected_from": best_dir}, open(os.path.join(args.outputs, "neumf_selected_meta.json"), "w"), indent=2)
            print(f"📦 Copied best checkpoint to {os.path.join(args.outputs, 'neumf.pt')}")

if __name__ == "__main__":
    main()
