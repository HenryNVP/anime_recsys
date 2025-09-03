import os, json, argparse, numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from .data import prepare_all
from .models.neumf import NeuMF

def train_neumf(
    data_dir="data", outputs="outputs",
    epochs=5, batch_size=8192, lr=3e-3, weight_decay=1e-6,
    emb_gmf=16, emb_mlp=32, neg_k=4, seed=42
):
    torch.manual_seed(seed); np.random.seed(seed)

    # Prepares and caches arrays under outputs/
    stats = prepare_all(
        data_dir=data_dir, out_dir=outputs,
        neg_k=neg_k, seed=seed
    )

    pack = np.load(os.path.join(outputs, "train_pairs.npz"))
    u = torch.tensor(pack["u"], dtype=torch.long)
    i = torch.tensor(pack["i"], dtype=torch.long)
    y = torch.tensor(pack["y"], dtype=torch.float32)

    dl = DataLoader(TensorDataset(u, i, y), batch_size=batch_size, shuffle=True)

    model = NeuMF(stats["num_users"], stats["num_items"], emb_gmf=emb_gmf, emb_mlp=emb_mlp)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCELoss()

    for ep in range(1, epochs+1):
        model.train(); running = 0.0
        for bu, bi, by in tqdm(dl, desc=f"Epoch {ep}"):
            bu, bi, by = bu.to(device), bi.to(device), by.to(device)
            pred = model(bu, bi)
            loss = bce(pred, by)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * len(by)
        print(f"Epoch {ep}: loss={running/len(y):.4f}")

    os.makedirs(outputs, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(outputs, "model.pt"))
    with open(os.path.join(outputs, "train_config.json"), "w") as f:
        json.dump({
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "emb_gmf": emb_gmf, "emb_mlp": emb_mlp,
            "neg_k": neg_k, "seed": seed
        }, f, indent=2)
    print("Saved model + config to", outputs)

def train_als(
    data_dir="data", outputs="outputs",
    factors=64, iterations=10, reg=0.01, seed=42
):
    import numpy as np
    import scipy.sparse as sp
    from .data import prepare_all
    from .models.als import ALSWrapper

    # Prepare (or reuse) cached arrays
    stats = prepare_all(data_dir=data_dir, out_dir=outputs, seed=seed)

    # Build implicit UI from train positives (y==1)
    pack = np.load(os.path.join(outputs, "train_pairs.npz"))
    tr_u, tr_i, tr_y = pack["u"], pack["i"], pack["y"]
    mask = tr_y == 1
    rows, cols = tr_u[mask], tr_i[mask]
    ui = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(stats["num_users"], stats["num_items"])
    ).tocsr()

    als = ALSWrapper(factors=factors, iterations=iterations, regularization=reg, random_state=seed)
    als.fit(ui)

    # Save factors for evaluation
    np.save(os.path.join(outputs, "als_user_factors.npy"), als.model.user_factors)
    np.save(os.path.join(outputs, "als_item_factors.npy"), als.model.item_factors)

    with open(os.path.join(outputs, "train_config.json"), "a") as f:
        pass  # optional: you can also write ALS config here if you like
    print("Saved ALS factors to", outputs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", type=str, default="neumf", choices=["neumf", "als"])
    # ... (keep the rest)

    # ALS args
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--reg", type=float, default=0.01)

    args = ap.parse_args()

    if args.trainer == "neumf":
        train_neumf(
            data_dir=args.data_dir, outputs=args.outputs,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            emb_gmf=args.emb_gmf, emb_mlp=args.emb_mlp,
            neg_k=args.neg_k, seed=args.seed
        )
    else:
        train_als(
            data_dir=args.data_dir, outputs=args.outputs,
            factors=args.factors, iterations=args.iterations, reg=args.reg, seed=args.seed
        )

if __name__ == "__main__":
    main()
