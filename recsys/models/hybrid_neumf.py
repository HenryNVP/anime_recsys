# recsys/models/hybrid_neumf.py
from __future__ import annotations
import torch
import torch.nn as nn

class HybridNeuMF(nn.Module):
    """
    Hybrid NeuMF:
      - GMF branch: elementwise product of user/item ID embeddings
      - MLP branch: concatenation of user_emb, item_emb, and item side features
      - Item features: genres, type, episodes, rating, members (from item_feats.npy)

    Output: sigmoid probability (implicit feedback prediction).
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        feat_dim: int,                # dimension of side features
        emb_gmf: int = 32,
        emb_mlp: int = 32,
        feat_proj: int = 32,
        mlp_layers: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.0,
    ):
        super().__init__()

        # --- ID embeddings
        self.user_gmf = nn.Embedding(num_users, emb_gmf)
        self.item_gmf = nn.Embedding(num_items, emb_gmf)
        self.user_mlp = nn.Embedding(num_users, emb_mlp)
        self.item_mlp = nn.Embedding(num_items, emb_mlp)

        # --- Project item side features into embedding space
        self.feat_proj = nn.Linear(feat_dim, feat_proj)

        # --- MLP tower
        mlp_in = emb_mlp + emb_mlp + feat_proj
        layers = []
        prev = mlp_in
        for h in mlp_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.mlp = nn.Sequential(*layers)

        # --- Final predictor
        self.out = nn.Linear(emb_gmf + prev, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        # Init embeddings small, linear layers with Xavier
        for emb in [self.user_gmf, self.item_gmf, self.user_mlp, self.item_mlp]:
            nn.init.normal_(emb.weight, std=0.01)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, u: torch.LongTensor, i: torch.LongTensor, item_feats: torch.FloatTensor):
        """
        u: [B] user indices
        i: [B] item indices
        item_feats: [num_items, feat_dim] full matrix; index with i
        """
        # --- GMF branch
        gu = self.user_gmf(u)          # [B, emb_gmf]
        gi = self.item_gmf(i)          # [B, emb_gmf]
        gmf = gu * gi                  # elementwise product

        # --- MLP branch
        mu = self.user_mlp(u)          # [B, emb_mlp]
        mi = self.item_mlp(i)          # [B, emb_mlp]
        fi = self.feat_proj(item_feats[i])  # [B, feat_proj]
        x = torch.cat([mu, mi, fi], dim=-1)
        mlp_out = self.mlp(x)          # [B, hidden]

        # --- Concatenate and predict
        z = torch.cat([gmf, mlp_out], dim=-1)
        return torch.sigmoid(self.out(z)).squeeze(-1)
