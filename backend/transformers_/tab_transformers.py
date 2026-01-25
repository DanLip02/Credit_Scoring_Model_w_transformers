# tab_transformer.py
import torch
import torch.nn as nn
# import torch.nn.functional as F

class CatEmbeddings(nn.Module):
    def __init__(self, cardinalities, emb_dim):
        super().__init__()
        # assert isinstance(emb_dim, int), f"emb_dim must be int, got {type(emb_dim)}, {emb_dim}"
        self.embs = nn.ModuleList([nn.Embedding(c+1, emb_dim, padding_idx=0) for c in cardinalities])  # +1 for unknown/pad

    def forward(self, x):
        # x: LongTensor [B, n_cat]
        out = []
        for i, emb in enumerate(self.embs):
            out.append(emb(x[:, i]))  # [B, D]
        # stack -> [B, n_cat, D]
        return torch.stack(out, dim=1)

class NumProj(nn.Module):
    def __init__(self, n_num, emb_dim):
        super().__init__()
        self.proj = nn.Linear(n_num, emb_dim)

    def forward(self, x):
        # x: [B, n_num]
        out = self.proj(x).unsqueeze(1)  # [B, 1, D]
        return out

class SimpleTransformer(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, seq_len, d_model]
        return self.encoder(x)

class TabTransformerModel(nn.Module):
    def __init__(self, cardinalities, n_num, emb_dim=34, nhead=4, n_layers=2, mlp_dim=64, dropout=0.1):
        super().__init__()

        emb_dim = int(emb_dim)
        assert isinstance(emb_dim, int), f"emb_dim must be int, got {type(emb_dim)}, {emb_dim}"

        assert all(isinstance(c, int) for c in cardinalities), f"all cardinalities must be int, got {cardinalities}"

        self.cat_emb = CatEmbeddings(cardinalities, emb_dim)
        self.num_proj = NumProj(n_num, emb_dim) if n_num>0 else None
        self.transformer = SimpleTransformer(d_model=emb_dim, nhead=nhead, num_layers=n_layers, dim_feedforward=mlp_dim, dropout=dropout)
        self.cls = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1)
        )

    def forward(self, cat_x, num_x=None):
        # cat_x: LongTensor [B, n_cat]; num_x: FloatTensor [B, n_num]
        cat_tokens = self.cat_emb(cat_x)           # [B, n_cat, D]
        if self.num_proj is not None and num_x is not None:
            num_token = self.num_proj(num_x)       # [B, 1, D]
            x = torch.cat([cat_tokens, num_token], dim=1)  # [B, n_cat+1, D]
        else:
            x = cat_tokens
        h = self.transformer(x)                    # [B, seq_len, D]
        pooled = h.mean(dim=1)                     # [B, D]
        logit = self.cls(pooled).squeeze(1)        # [B]
        return torch.sigmoid(logit)
