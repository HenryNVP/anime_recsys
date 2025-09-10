# scripts/tune_neumf.py
from __future__ import annotations
import os, json, argparse, shutil
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from recsys.data import load_splits, build_eval_candidates
from recsys.metrics import hr_ndcg_at_k
from recsys.models.neumf import NeuMF
try:
    from recsys.models.hybrid_neumf import HybridNeuMF
    HAS_HYBRID = True
except Exception:
    HAS_HYBRID = False


# ------------------------ configs ------------------------

@dataclass(frozen=True)
class NeuMFConfig:
    model_name: str           # "neumf" or "hybrid"
    emb_gmf: int
    emb_mlp: int
    mlp_layers: Tuple[int, ...]
    lr: float
    neg_k: int
    feat_proj: Optional[int] = None  # only for hybrid

    def slug(self) -> str:
        layers = "-".join(map(str, self.mlp_layers))
        if self.model_name == "hybrid" and self.feat_proj is not None:
            return f"{self.model_name}_gmf{self.emb_gmf}_mlp{self.emb_mlp}_feat{self.feat_proj}_layers{layers}_lr{self.lr}_neg{self.neg_k}"
        return f"{self.model_name}_gmf{self.emb_gmf}_mlp{self.emb_mlp}_layers{layers}_lr{self.lr}_neg{self.neg_k}"


# ------------------------ helpers ------------------------

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
def _eval_logits_ranking(model: nn.Module, users_arr: np.ndarray, cands: np.ndarray,
                         device: torch.device, extra_item_feats: Optional[torch.Tensor],
                         k_eval: int):
    scores = np.zeros_like(cands, dtype=np.float32)
    for r in range(cands.shape[0]):
        u = int(users_arr[r]); cand = cands[r]
        uu = torch.full((len(cand),), u, dtype=torch.long, device=device)
        ii = torch.tensor(cand, dtype=torch.long, device=device)
        if extra_item_feats is None:
            scores[r] = model(uu, ii).detach().cpu().numpy()
        else:
            scores[r] = model(uu, ii, extra_item_feats).detach().cpu().numpy()
    return hr_ndcg_at_k(scores, k_eval)


# ------------------------ one run ------------------------

def train_one_config(
    cfg: NeuMFConfig,
    data_dir: str, outputs: str,
    seed: int, epochs: int, batch_size: int, weight_decay: float,
    k_eval: int, patience: int, num_workers: int, amp: bool,
    resume: bool
):
    # dirs
    tune_root = os.path.join(outputs, f"{cfg.model_name}_tuning"); os.makedirs(tune_root, exist_ok=True)
    run_dir = os.path.join(tune_root, cfg.slug()); os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "last_ckpt.pt")
    best_meta_path = os.path.join(run_dir, "best_meta.json")

    # seeds
    torch.manual_seed(seed); np.random.seed(seed)

    # data & candidates
    train_df, val_df, test_df, stats = load_splits(data_dir)
    ensure_candidates(train_df, val_df, test_df, stats, outputs, seed)

    # cached pairs per neg_k
    u, i, y = build_pointwise_pairs_cached(train_df, stats["num_items"], cfg.neg_k, seed, outputs)
    dl = DataLoader(TensorDataset(u,i,y), batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True, drop_last=False)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    if cfg.model_name == "neumf":
        model = NeuMF(stats["num_users"], stats["num_items"],
                      emb_gmf=cfg.emb_gmf, emb_mlp=cfg.emb_mlp, mlp_layers=cfg.mlp_layers).to(device)
        extra_feats = None
    elif cfg.model_name == "hybrid":
        if not HAS_HYBRID:
            raise RuntimeError("HybridNeuMF not found. Ensure recsys/models/hybrid_neumf.py exists.")
        # require item_feats created by your features pipeline (saved to outputs/item_feats.npy)
        feats_path = os.path.join(outputs, "item_feats.npy")
        if not os.path.exists(feats_path):
            raise FileNotFoundError(f"Missing item features at {feats_path}. Run your features pipeline first.")
        item_feats = torch.tensor(np.load(feats_path), dtype=torch.float32).to(device, non_blocking=True)

        model = HybridNeuMF(stats["num_users"], stats["num_items"],
                            feat_dim=item_feats.shape[1],
                            emb_gmf=cfg.emb_gmf, emb_mlp=cfg.emb_mlp,
                            feat_proj=(cfg.feat_proj or 32),
                            mlp_layers=cfg.mlp_layers).to(device)
        extra_feats = item_feats
    else:
        raise ValueError("model_name must be 'neumf' or 'hybrid'")

    # optimizer / loss (logits + AMP-safe)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    # resume
    best = {"epoch": 0, "val_hr": 0.0, "val_ndcg": 0.0}
    history = []
    start_epoch = 1
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
    users_val = np.load(os.path.join(outputs, "val_users.npy"))
    cands_val = np.load(os.path.join(outputs, "val_cands.npy"))
    for ep in range(start_epoch, epochs+1):
        model.train(); running = 0.0
        for bu, bi, by in tqdm(dl, desc=f"[{cfg.slug()}] epoch {ep}/{epochs}"):
            bu, bi, by = bu.to(device, non_blocking=True), bi.to(device, non_blocking=True), by.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=amp):
                logits = model(bu, bi) if extra_feats is None else model(bu, bi, extra_feats)
                loss = bce(logits, by)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            running += loss.item() * len(by)

        train_bce = running / len(y)
        model.eval()
        val_hr, val_ndcg = _eval_logits_ranking(model, users_val, cands_val, device, extra_feats, k_eval)
        history.append({"epoch": ep, "train_bce": float(train_bce),
                        f"val_hr@{k_eval}": float(val_hr), f"val_ndcg@{k_eval}": float(val_ndcg)})

        # checkpoint (for resume)
        torch.save({"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict() if amp else None,
                    "best": best, "history": history}, ckpt_path)

        # early stopping on val NDCG
        if val_ndcg > best["val_ndcg"]:
            best = {"epoch": ep, "val_hr": float(val_hr), "val_ndcg": float(val_ndcg)}
            ckpt_name = "neumf.pt" if cfg.model_name == "neumf" else "hybrid_neumf.pt"
            torch.save(model.state_dict(), os.path.join(run_dir, ckpt_name))
            json.dump({"config": asdict(cfg), "best": best}, open(best_meta_path,"w"), indent=2)
            print("  ✅ improved; saved best")
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"⏹ early stopping (patience={patience})")
                break

    return {"config": cfg, "best": best, "run_dir": run_dir}


# ------------------------ main (grid) ------------------------

def main():
    ap = argparse.ArgumentParser("Tune NeuMF/Hybrid with early stopping; simple skip/resume")
    ap.add_argument("--model", default="neumf", choices=["neumf", "hybrid"])
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

    # Focus on embedding sizes and MLP depth/width
    ap.add_argument("--emb_gmf_grid", type=str, default="32,64,128")
    ap.add_argument("--emb_mlp_grid", type=str, default="32,64,128")
    ap.add_argument("--mlp_grid", type=str, default="256-128-64,512-256-128")
    ap.add_argument("--lr_grid", type=str, default="0.003,0.001")
    ap.add_argument("--negk_grid", type=str, default="4")

    # Only for hybrid
    ap.add_argument("--feat_proj_grid", type=str, default="32,64")

    ap.add_argument("--summary_csv", default=None)
    args = ap.parse_args()

    # parse grids
    emb_gmf_grid = [int(x) for x in args.emb_gmf_grid.split(",") if x.strip()]
    emb_mlp_grid = [int(x) for x in args.emb_mlp_grid.split(",") if x.strip()]
    mlp_grid = [tuple(int(y) for y in s.split("-") if y.strip()) for s in args.mlp_grid.split(",") if s.strip()]
    lr_grid = [float(x) for x in args.lr_grid.split(",") if x.strip()]
    negk_grid = [int(x) for x in args.negk_grid.split(",") if x.strip()]
    feat_proj_grid = [int(x) for x in args.feat_proj_grid.split(",") if x.strip()] if args.model == "hybrid" else [None]

    # configs
    configs: List[NeuMFConfig] = [
        NeuMFConfig(args.model, eg, em, ml, lr, nk, fp)
        for eg in emb_gmf_grid
        for em in emb_mlp_grid
        for ml in mlp_grid
        for lr in lr_grid
        for nk in negk_grid
        for fp in feat_proj_grid
    ]
    print(f"Total configs: {len(configs)} for model={args.model}")

    rows = []
    for cfg in configs:
        run_dir = os.path.join(args.outputs, f"{args.model}_tuning", cfg.slug())
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
    summary_name = args.summary_csv or (f"{args.model}_tuning_summary.csv")
    out_csv = os.path.join(args.outputs, summary_name)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved summary CSV to {out_csv}")

    # copy best checkpoint up one level
    if len(df):
        best_dir = df.iloc[0]["run_dir"]
        if args.model == "neumf":
            src = os.path.join(best_dir, "neumf.pt")
            dst = os.path.join(args.outputs, "neumf.pt")
            meta_dst = os.path.join(args.outputs, "neumf_selected_meta.json")
        else:
            src = os.path.join(best_dir, "hybrid_neumf.pt")
            dst = os.path.join(args.outputs, "hybrid_neumf.pt")
            meta_dst = os.path.join(args.outputs, "hybrid_selected_meta.json")

        if os.path.exists(src):
            shutil.copy2(src, dst)
            json.dump({"selected_from": best_dir}, open(meta_dst, "w"), indent=2)
            print(f"📦 Copied best checkpoint to {dst}")

if __name__ == "__main__":
    main()
