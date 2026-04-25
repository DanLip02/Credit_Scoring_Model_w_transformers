# tab_transformer.py
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
import numpy as np
# import torch.nn.functional as F


def compute_feature_stats(X: np.ndarray) -> np.ndarray:
    """
    Input:  X [N, n_features]
    Output: stats [n_features, 8] — [mean, std, skew, kurt, min, max, median, iqr]
    """
    try:
        use_scipy = True
    except ImportError:
        use_scipy = False

    result = []
    for i in range(X.shape[1]):
        col = X[:, i].astype(np.float32)
        col_clean = col[~np.isnan(col)]
        if len(col_clean) == 0:
            col_clean = np.zeros(1, dtype=np.float32)
        q75, q25 = np.percentile(col_clean, [75, 25])
        skew = float(scipy_stats.skew(col_clean)) if use_scipy else 0.0
        kurt = float(scipy_stats.kurtosis(col_clean)) if use_scipy else 0.0
        result.append([
            float(np.mean(col_clean)),
            float(np.std(col_clean) + 1e-8),
            skew, kurt,
            float(np.min(col_clean)),
            float(np.max(col_clean)),
            float(np.median(col_clean)),
            float(q75 - q25),
        ])
    return np.array(result, dtype=np.float32)  # [n_features, 8]


class FeatureStatEmbedder(nn.Module):
    """
    Проецирует статистики фичи [8] → [emb_dim].
    Using in stat_based_init for smarter init. of  NumProj/CatEmbeddings.
    """

    def __init__(self, emb_dim: int, n_stats: int = 8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_stats, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.proj(stats)  # [n_features, emb_dim]

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
        # self.proj = nn.Linear(n_num, emb_dim)
        self.proj = nn.ModuleList([nn.Linear(1, emb_dim) for _ in range(n_num)])

    def forward(self, x):
        # x: [B, n_num]
        tokens = []
        # out = self.proj(x).unsqueeze(1)  # [B, 1, D]   -> All numeric features to one

        for i, layer in enumerate(self.proj):
            token = layer(x[:, i].unsqueeze(1))
            tokens.append(token)
        return torch.stack(tokens, dim=1)

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
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))  #cls token - not mean pooling, but with weights.
        self.cls = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1)
        )

    def forward(self, cat_x, num_x=None):
        # cat_x: LongTensor [B, n_cat]; num_x: FloatTensor [B, n_num]
        tokens = []

        # Categorical token if numeric exists
        if cat_x is not None and len(self.cat_emb.embs) > 0:
            cat_tokens = self.cat_emb(cat_x)  # [B, n_cat, D]
            tokens.append(cat_tokens)

        # Numeric token if numeric exists
        if self.num_proj is not None and num_x is not None:
            num_token = self.num_proj(num_x)  # [B, 1, D]
            tokens.append(num_token)

        assert len(tokens) > 0, "No catigorical, No numerical!"

        x = torch.cat(tokens, dim=1)  # [B, seq_len, D]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        h = self.transformer(x)  # [B, seq_len, D]
        # pooled = h.mean(dim=1)  # [B, D]
        pooled = h[:, 0]
        logit = self.cls(pooled).squeeze(1)  # [B]
        return torch.sigmoid(logit)

    def freeze_backbone(self, mode: str = "full"):
        """
        freeze backbone for finetune on new dataset.

        mode:
            "full"       — freeze all transformer + cls_token
                           learn only cat_emb, num_proj, cls (head).
                           When dataset is small.

            "last_layer" — freeze all layers of transformer (last is not).
                           Balance between adaptation and stability.
                           When dataset is normal.

            "none"       — Full finetune.
                           Tha task is same (big dataset).
        """
        if mode == "full":
            for p in self.transformer.parameters():
                p.requires_grad = False
            self.cls_token.requires_grad = False

        elif mode == "last_layer":

            layers = list(self.transformer.encoder.layers)
            for layer in layers[:-1]:
                for p in layer.parameters():
                    p.requires_grad = False
            # Last layer and cls_token are still ready for learning
            for p in layers[-1].parameters():
                p.requires_grad = True
            self.cls_token.requires_grad = True

        elif mode == "none":
            pass

        else:
            raise ValueError(f"Unknown freeze mode: {mode}. Use 'full', 'last_layer' or 'none'.")

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[freeze_backbone mode={mode}] "
              f"Parameters to learn: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.1f}%)")

    def unfreeze_backbone(self):
        """Unfreeze all."""
        for p in self.parameters():
            p.requires_grad = True
        print("[unfreeze_backbone] All parameters are unfrozen.")

    def get_backbone_state_dict(self) -> dict:
        """
        Return only weights backbone (transformer + cls_token).
        Save before transfer on new dataset.
        Weight of cat_emb, num_proj, cls — specific, no transfer there.
        """
        return {
            k: v for k, v in self.state_dict().items()
            if k.startswith("transformer.") or k == "cls_token"
        }

    def load_backbone_state_dict(self, backbone_state: dict):
        """
        Load all weights backbone from dict (get_backbone_state_dict).
        (input/head) are random — again learning.
        """
        missing, unexpected = self.load_state_dict(backbone_state, strict=False)
        loaded = len(backbone_state)
        expected_missing = [k for k in missing
                            if any(k.startswith(p) for p in ("cat_emb.", "num_proj.", "cls."))]
        real_missing = [k for k in missing if k not in expected_missing]
        if real_missing:
            print(f"[load_backbone] not found: {real_missing}")
        print(f"[load_backbone] Load {loaded} backbone-wights. "
              f"Input/head init again.")

