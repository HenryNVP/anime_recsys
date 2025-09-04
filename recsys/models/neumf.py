from typing import Tuple
import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, num_users: int, num_items: int,
                 emb_gmf: int = 16, emb_mlp: int = 32,
                 mlp_layers: Tuple[int, ...] = (64, 32, 16)):
        super().__init__()
        # GMF branch
        self.user_gmf = nn.Embedding(num_users, emb_gmf)
        self.item_gmf = nn.Embedding(num_items, emb_gmf)
        # MLP branch
        self.user_mlp = nn.Embedding(num_users, emb_mlp)
        self.item_mlp = nn.Embedding(num_items, emb_mlp)
        layers, in_dim = [], emb_mlp * 2
        for h in mlp_layers:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        # Fusion + output
        self.fc = nn.Linear(emb_gmf + mlp_layers[-1], 1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, u, i):
        g = self.user_gmf(u) * self.item_gmf(i)           # GMF
        m = torch.cat([self.user_mlp(u), self.item_mlp(i)], dim=1)
        m = self.mlp(m)                                    # MLP
        x = torch.cat([g, m], dim=1)
        return torch.sigmoid(self.fc(x).squeeze(-1))
