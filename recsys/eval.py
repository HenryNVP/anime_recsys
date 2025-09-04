import os, argparse, pickle, json
import numpy as np
import torch

from .metrics import hr_ndcg_at_k
from .models.neumf import NeuMF
from .models.knn import UserKNN

def _load_shapes(out_dir: str):
    with open(os.path.join(out_dir, "mappings.pkl"), "rb") as f:
        m = pickle.load(f)
    return m["num_users"], m["num_items"]

def eval_neumf(out_dir: str, split: str, k: int):
    num_users, num_items = _load_shapes(out_dir)
    model = NeuMF(num_users, num_items)
    model.load_state_dict(torch.load(os.path.join(out_dir, "model.pt"), map_location="cpu"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))

    scores = np.zeros_like(cands, dtype=float)
    with torch.no_grad():
        for r in range(cands.shape[0]):
            u = int(users[r]); cand = cands[r]
            uu = torch.full((len(cand),), u, dtype=torch.long, device=device)
            ii = torch.tensor(cand, dtype=torch.long, device=device)
            scores[r] = model(uu, ii).detach().cpu().numpy()
    return hr_ndcg_at_k(scores, k=k)



def eval_userknn(out_dir: str, split: str = "val", k: int = 10):
    pack = np.load(os.path.join(out_dir, "userknn_model.npz"), allow_pickle=True)
    user_sim, ui = pack["user_sim"], pack["ui"].item() if pack["ui"].shape == () else pack["ui"]

    model = UserKNN()
    model.user_sim, model.ui = user_sim, ui

    users = np.load(os.path.join(out_dir, f"{split}_users.npy"))
    cands = np.load(os.path.join(out_dir, f"{split}_cands.npy"))

    scores = np.zeros_like(cands, dtype=float)
    for r in range(cands.shape[0]):
        u, cand = int(users[r]), cands[r]
        scores[r] = model.score_user_candidates(u, cand)

    return hr_ndcg_at_k(scores, k=k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=str, default="outputs")
    ap.add_argument("--model", type=str, default="neumf", choices=["neumf", "userknn"])
    ap.add_argument("--split", type=str, default="val", choices=["val", "test"])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    if args.model == "neumf":
        hr, ndcg = eval_neumf(args.outputs, args.split, args.k)
    else:
        hr, ndcg = eval_userknn(args.outputs, args.split, args.k)

    os.makedirs("results", exist_ok=True)
    out = {"model": args.model, "split": args.split, f"HR@{args.k}": hr, f"NDCG@{args.k}": ndcg}
    with open("results/metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
