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

def finetune_tabtransformer(
    cat_train, num_train, y_train,
    cat_val,   num_val,   y_val,
    cardinalities: list,
    TT: dict,
    backbone_checkpoint: str,
    freeze_mode: str = "last_layer",
    device: str = "cpu",
):
    """
    Stage B — finetune on new dataset

    Algorithm:
      1. Create TabTransformerModel with new feature structure of dataset B.
         (cat_emb и num_proj — new / random).
      2. Load backbone (transformer + cls_token) from Stage A.
         Input/head are random.
      3. Freeze backbone with freeze_mode.
      4.Using 2 learning rate:
           backbone: lr * 0.1
           input + head:lr
      5. Save the best with AUC.

    freeze_mode: "full" | "last_layer" | "none"
        "full"       → learn only new cat_emb/num_proj/cls
        "last_layer" → last layer of transformer
        "none"       → full finetune
    backbone_checkpoint: path to .pth file with weights from Stage A
    """
    from .tab_transformers import TabTransformerModel

    print(f"\n{'='*55}")
    print(f"FINETUNE (freeze_mode={freeze_mode!r})")
    print(f"  Backbone: {backbone_checkpoint}")
    print(f"  Train: {len(y_train)} | Val: {len(y_val)}")
    print(f"{'='*55}")

    n_num = num_train.shape[1] if num_train is not None else 0
    model = TabTransformerModel(
        cardinalities=cardinalities,
        n_num=n_num,
        emb_dim=TT["embed_dim"],
        nhead=TT["n_heads"],
        n_layers=TT["n_layers"],
        mlp_dim=TT["mlp_dim"],
        dropout=TT["dropout"],
    ).to(device)

    # load backbone
    # backbone_state = torch.load(backbone_checkpoint, map_location=device)
    loaded = torch.load(backbone_checkpoint, map_location=device, weights_only=False)
    backbone_state = loaded.state_dict() if hasattr(loaded, 'state_dict') else loaded

    backbone_keys = {k: v for k, v in backbone_state.items()
                     if k.startswith("transformer.") or k == "cls_token"}
    model.load_backbone_state_dict(backbone_keys)

    # freeze necessary layers
    model.freeze_backbone(mode=freeze_mode)

    # two lr: small for backbone, basic for new layers
    backbone_params = [p for n, p in model.named_parameters()
                       if (n.startswith("transformer.") or n == "cls_token")
                       and p.requires_grad]
    new_params = [p for n, p in model.named_parameters()
                  if not (n.startswith("transformer.") or n == "cls_token")]

    param_groups = [{"params": new_params, "lr": TT["lr"]}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": TT["lr"] * 0.1})

    opt = torch.optim.AdamW(param_groups, weight_decay=TT.get("weight_decay", 0.01))

    # Loss with balance of weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weight = TT.get("class_weight", "balanced")
    if class_weight == "balanced":
        weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        pos_weight = torch.tensor([weights[1] / weights[0]], dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    train_ds = TabDataset(cat_train, num_train, y_train)
    val_ds   = TabDataset(cat_val,   num_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=TT["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=TT["batch_size"], shuffle=False)

    best_auc = 0.0
    patience = TT.get("early_stopping_patience", 10)
    patience_counter = 0

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(BASE_DIR, "..", "..", "models", "transformers_")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"tabtransformer_finetune_{freeze_mode}.pth")

    for epoch in range(TT["epochs"]):
        train_loss = train_loop(model, train_loader, opt, loss_fn, device)
        ys_val, preds_val = eval_loop(model, val_loader, device)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(ys_val, preds_val)

        marker = ""
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            marker = " ← best"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1:3d}/{TT['epochs']} | "
              f"loss={train_loss:.4f} | val_auc={auc:.4f}{marker}")

        if patience_counter >= patience:
            print(f"  Early stopping on epoch {epoch+1}")
            break

    print(f"\n Best AUC={best_auc:.4f}")
    print(f"Checkpoint: {save_path}")

    # load the best wights to the model
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, save_path


def adapt_to_new_dataset(
    cat_data, num_data, y_data,
    cardinalities: list,
    TT: dict,
    backbone_checkpoint: str,
    mode: str = "zero_shot",
    adapt_epochs: int = 5,
    device: str = "cpu",
):
    """
    mode:
        "zero_shot"  — backbone load, input/head random.
                       No finetune. Fast but not accurate.
                       table patterns, not actually features.

        "proj_adapt" — backbone is frozen, learn only new
                       cat_emb + num_proj + cls_head by adapt_epochs.
                       Have enough classified target.

    Return: (model, predict_proba[N, 2])
    """
    from .tab_transformers import TabTransformerModel

    mode_label = "Zero-shot" if mode == "zero_shot" else f"Proj-adapt ({adapt_epochs} эпох)"
    print(f"\n{'='*55}")
    print(f"ADAPT TO C ({mode_label})")
    print(f"  Backbone: {backbone_checkpoint}")
    print(f"  Data: {len(y_data) if y_data is not None else '—'}")
    print(f"{'='*55}")

    n_num = num_data.shape[1] if num_data is not None else 0
    model = TabTransformerModel(
        cardinalities=cardinalities,
        n_num=n_num,
        emb_dim=TT["embed_dim"],
        nhead=TT["n_heads"],
        n_layers=TT["n_layers"],
        mlp_dim=TT["mlp_dim"],
        dropout=TT["dropout"],
    ).to(device)

    # backbone_state = torch.load(backbone_checkpoint, map_location=device)

    loaded = torch.load(backbone_checkpoint, map_location=device, weights_only=False)
    # Если MLflow сохранил модель целиком — достаём state_dict
    backbone_state = loaded.state_dict() if hasattr(loaded, 'state_dict') else loaded

    backbone_keys = {k: v for k, v in backbone_state.items()
                     if k.startswith("transformer.") or k == "cls_token"}
    model.load_backbone_state_dict(backbone_keys)

    if mode == "zero_shot":
        # only inference
        model.freeze_backbone("full")
        print("  Zero-shot:")

    elif mode == "proj_adapt":
        # Backbone is frozen, learn only input+head
        model.freeze_backbone("full")
        assert y_data is not None, "proj_adapt требует y_data (метки для C)"

        adapt_params = (
            list(model.cat_emb.parameters()) +
            (list(model.num_proj.parameters()) if model.num_proj else []) +
            list(model.cls.parameters())
        )
        opt = torch.optim.Adam(adapt_params, lr=TT["lr"] * 0.3)
        loss_fn = torch.nn.BCELoss()

        adapt_ds = TabDataset(cat_data, num_data, y_data)
        adapt_loader = DataLoader(adapt_ds, batch_size=TT["batch_size"], shuffle=True)

        for epoch in range(adapt_epochs):
            model.train()
            total_loss = 0.0
            for batch in adapt_loader:
                cat = batch.get("cat", None)
                num = batch.get("num", None)
                y = batch["y"].to(device)
                if cat is not None:
                    cat = cat.to(device)
                if num is not None:
                    num = num.to(device)
                opt.zero_grad()
                out = model(cat, num)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(y)
            avg = total_loss / len(adapt_loader.dataset)
            print(f"  Adapt epoch {epoch+1}/{adapt_epochs} | loss={avg:.4f}")

    # Inference
    model.eval()
    ds = TabDataset(cat_data, num_data, y=None)
    loader = DataLoader(ds, batch_size=TT["batch_size"], shuffle=False)
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            cat = batch.get("cat", None)
            num = batch.get("num", None)
            if cat is not None:
                cat = cat.to(device)
            if num is not None:
                num = num.to(device)
            preds = model(cat, num)
            all_preds.append(preds.cpu().numpy())

    probas = np.concatenate(all_preds)
    return model, np.column_stack([1 - probas, probas])