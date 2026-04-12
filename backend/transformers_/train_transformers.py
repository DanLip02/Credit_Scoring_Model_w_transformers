# train_transformer.py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from .tab_transformers import TabTransformerModel
# from config import TT
import os

# TT = {
#     "embed_dim": 32,
#     "n_heads": 4,
#     "n_layers": 2,
#     "mlp_dim": 64,
#     "dropout": 0.1,
#     "lr": 1e-3,
#     "batch_size": 128,
#     "epochs": 20,
# }

class TabDataset(Dataset):
    def __init__(self, cat, num, y=None):
        self.cat = cat.astype("int64") if cat is not None else None #or it can be a prblem here
        self.num = num.astype("float32") if num is not None else None
        self.y = y.astype("float32") if y is not None else None

    def __len__(self):
        if self.cat is not None:
            return len(self.cat)

        if self.num is not None:
            return len(self.num)

        return len(self.y)

    def __getitem__(self, idx):
        item = {}
        if self.cat is not None:  # ✅
            item["cat"] = torch.from_numpy(self.cat[idx])
        if self.num is not None:
            item["num"] = torch.from_numpy(self.num[idx])
        if self.y is not None:
            item["y"] = torch.tensor(self.y[idx], dtype=torch.float32)
        return item

def train_loop(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        cat = batch["cat"].to(device) if "cat" in batch else None
        num = batch["num"].to(device) if "num" in batch else None
        y = batch["y"].to(device)
        opt.zero_grad()
        out = model(cat, num)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)

def eval_loop(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            cat = batch["cat"].to(device) if "cat" in batch else None
            num = batch["num"].to(device) if "num" in batch else None
            y = batch["y"].to(device)
            out = model(cat, num)
            ys.append(y.cpu().numpy())
            preds.append(out.cpu().numpy())
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    return ys, preds

def fit_tabtransformer(cat_train, num_train, y_train, cat_val, num_val, y_val, cardinalities, TT, device="cpu"):
    np.random.seed(42)

    class_weight = TT.get("class_weight", None)
    weight_decay = TT.get("weight_decay", 0.01)

    model = TabTransformerModel(cardinalities=cardinalities, n_num=num_train.shape[1] if num_train is not None else 0,
                                emb_dim=TT["embed_dim"], nhead=TT["n_heads"], n_layers=TT["n_layers"], mlp_dim=TT["mlp_dim"], dropout=TT["dropout"])
    model.to(device)

    if class_weight is not None:
        if isinstance(class_weight, list):

            pos_weight = torch.tensor([class_weight[1] / class_weight[0]], dtype=torch.float32).to(device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif class_weight == "balanced":
            from sklearn.utils.class_weight import compute_class_weight
            weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            pos_weight = torch.tensor([weights[1] / weights[0]], dtype=torch.float32).to(device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    #loss_fn = nn.BCELoss() # todo add BCEloss optional
    # opt = optim.Adam(model.parameters(), lr=TT["lr"]) #todo weight_decay is None or not
    opt = optim.AdamW(model.parameters(), lr=TT["lr"], weight_decay=weight_decay)

    train_ds = TabDataset(cat_train, num_train, y_train)
    val_ds = TabDataset(cat_val, num_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=TT["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TT["batch_size"], shuffle=False)

    total_steps = TT["epochs"] * len(train_loader)
    warmup_steps = int(total_steps * TT.get("warmup_ratio", 0.1)) if TT.get("warmup_ratio", 0.1) > 0 else 0

    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
    else:
        scheduler = None

    # train_ds = TabDataset(cat_train, num_train, y_train)
    # val_ds = TabDataset(cat_val, num_val, y_val)
    # train_loader = DataLoader(train_ds, batch_size=TT["batch_size"], shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=TT["batch_size"], shuffle=False)

    best_auc = 0.0
    for epoch in range(TT["epochs"]):
        train_loss = train_loop(model, train_loader, opt, loss_fn, device)

        if scheduler is not None and epoch < warmup_steps // len(train_loader):
            scheduler.step()

        ys_val, preds_val = eval_loop(model, val_loader, device)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(ys_val, preds_val)
        print(f"Epoch {epoch+1}/{TT['epochs']} train_loss={train_loss:.4f} val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            # torch.save(model.state_dict(), os.path.join("../..", "models", "transformers_", "tabtransformer_best.pth"))
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))

            save_dir = os.path.join(BASE_DIR, "..", "..", "models", "transformers_")
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, "tabtransformer_best.pth")

            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    return model
