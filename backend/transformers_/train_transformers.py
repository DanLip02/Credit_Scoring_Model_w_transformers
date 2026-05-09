# train_transformer.py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from .tab_transformers import TabTransformerModel, compute_feature_stats, FeatureStatEmbedder
import os
import time

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
        if self.cat is not None:
            item["cat"] = torch.from_numpy(self.cat[idx])
        if self.num is not None:
            item["num"] = torch.from_numpy(self.num[idx])
        if self.y is not None:
            item["y"] = torch.tensor(self.y[idx], dtype=torch.float32)
        return item


def merge_embeddings(source_embs_A, source_embs_B,
                     match_indices,
                     source_col_names_A,
                     source_col_names_B,
                     alpha=0.7):
    result = {}

    for key in ['num', 'cat']:
        if key not in source_embs_A or key not in source_embs_B:
            if key in source_embs_A:
                result[key] = source_embs_A[key].clone()
            continue

        embs_A = source_embs_A[key]  # [n_A, emb_dim]
        embs_B = source_embs_B[key]  # [n_B, emb_dim]
        merged = embs_A.clone()

        # avarage
        matched_B = set()
        for i_B, i_A in match_indices[key].items():
            if i_B < embs_B.shape[0] and i_A < embs_A.shape[0]:
                merged[i_A] = alpha * embs_B[i_B] + (1 - alpha) * embs_A[i_A]
                matched_B.add(i_B)
                print(f"  [merge] {source_col_names_B[i_B]:25s} ↔ "
                      f"{source_col_names_A[i_A]:25s} (alpha={alpha})")

        # — feature only in new - add
        new_embs = []
        for i_B in range(embs_B.shape[0]):
            if i_B not in matched_B:
                new_embs.append(embs_B[i_B])
                col_name = source_col_names_B[i_B] if i_B < len(source_col_names_B) else f"feat_{i_B}"
                print(f"  [new]   {col_name} added from new stage")

        if new_embs:
            new_tensor = torch.stack(new_embs)
            merged = torch.cat([merged, new_tensor], dim=0)

        result[key] = merged

    print(f"\n[merge_embeddings] num: {source_embs_A.get('num', torch.tensor([])).shape[0]} + "
          f"новых {len([i for i in range(source_embs_B.get('num', torch.tensor([])).shape[0]) if i not in match_indices.get('num', {}).keys()])} "
          f"= {result.get('num', torch.tensor([])).shape[0]}")

    return result

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

        #todo after finetune with transfer learning NaN in preds_val -
        # after this should not be Nan
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

            #todo check why NaN in transfer learning:
            # if num is not None:
            #     print(f"num NaN: {torch.isnan(num).sum().item()}, inf: {torch.isinf(num).sum().item()}")
            # if cat is not None:
            #     print(f"cat NaN: {torch.isnan(cat.float()).sum().item()}")

            out = model(cat, num)

            # print(f"out NaN: {torch.isnan(out).sum().item()}")
            # print(f"out sample: {out[:5]}")
            # break

            ys.append(y.cpu().numpy())
            preds.append(out.cpu().numpy())
    ys = np.concatenate(ys)
    preds = np.concatenate(preds)
    return ys, preds

def fit_tabtransformer(cat_train, num_train, y_train, cat_val, num_val, y_val, cardinalities, TT, device="cpu"):
    np.random.seed(42)
    print(TT)
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

            print(f"class_weight param = {class_weight!r}")
            print(f"pos_weight value   = {pos_weight.item():.4f}")
            print(f"y_train counts     = {np.bincount(y_train.astype(int))}")

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

    warmup_epochs = int(TT["epochs"] * TT.get("warmup_ratio", 0.1))

    if warmup_steps > 0:
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
    else:
        scheduler_warmup = None

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=TT["epochs"] - warmup_epochs,
        eta_min=1e-6
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_steps]
    ) if scheduler_warmup is not None else scheduler_cosine

    # train_ds = TabDataset(cat_train, num_train, y_train)
    # val_ds = TabDataset(cat_val, num_val, y_val)
    # train_loader = DataLoader(train_ds, batch_size=TT["batch_size"], shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=TT["batch_size"], shuffle=False)

    best_auc = 0.0
    for epoch in range(TT["epochs"]):
        train_loss = train_loop(model, train_loader, opt, loss_fn, device)

        if scheduler is not None:
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

            feature_embs = _extract_feature_embeddings(model, device)
            emb_path = save_path.replace(".pth", "_feature_embs.pth")
            torch.save(feature_embs, emb_path)
            print(f"Feature embeddings saved to {emb_path}")

    return model


def _extract_feature_embeddings(model, device="cpu") -> dict:
    """
    extract feature embedding from model's weight .
    Save with Stage A, load with Stage B/C для alignment_loss.

    Return dict:
        "num": [n_num, emb_dim]  — average weights NumProj
        "cat": [n_cat, emb_dim]  — average weights CatEmbeddings
    """
    result = {}
    with torch.no_grad():
        if model.num_proj is not None:
            num_embs = torch.stack([
                proj.weight.mean(dim=1)  # [emb_dim]
                for proj in model.num_proj.proj
            ])  # [n_num, emb_dim]
            result["num"] = num_embs.to(device)

        if len(model.cat_emb.embs) > 0:
            cat_embs = torch.stack([
                emb.weight.mean(dim=0)  # [emb_dim]
                for emb in model.cat_emb.embs
            ])  # [n_cat, emb_dim]
            result["cat"] = cat_embs.to(device)

    return result

def _extract_feature_embeddings_grad(model, device):
    result = {}
    if model.num_proj is not None:
        num_embs = torch.stack([
            proj.weight.mean(dim=1)
            for proj in model.num_proj.proj
        ])
        result["num"] = num_embs.to(device)

    if len(model.cat_emb.embs) > 0:
        cat_embs = torch.stack([
            emb.weight.mean(dim=0)
            for emb in model.cat_emb.embs
        ])
        result["cat"] = cat_embs.to(device)

    return result

def alignment_loss(
        embs_source: dict,  # from _extract_feature_embeddings model A
        embs_target: dict,  # from _extract_feature_embeddings other models
        lambda_align: float = 0.1,
        margin: float = 0.3,
        match_indices: dict = None
) -> torch.Tensor:
    """
    Contrastive Alignment Loss between feature 's embeddings of datasets A and other.

    Algo:
      - for each feature, target try to find the closest feature source.
      - if simmilarity is high (> 0.5) — make it closer.
      - if simmilarity is low (< margin) — regularization.

    the same feature still close to each other in latent space.
    """
    import torch.nn.functional as F

    loss = torch.tensor(0.0, requires_grad=True)
    n_pairs = 0

    for key in ["num", "cat"]:
        if key not in embs_source or key not in embs_target:
            continue

        src = embs_source[key]  # [n_source, D]
        tgt = embs_target[key]  # [n_target, D]

        for i in range(tgt.shape[0]):
            if match_indices and key in match_indices and i in match_indices[key]:
                j = match_indices[key][i]
                if j < src.shape[0]:
                    sim = F.cosine_similarity(
                        tgt[i].unsqueeze(0),
                        src[j].unsqueeze(0)
                    ).squeeze()
                    loss = loss + (1.0 - sim)
                    n_pairs += 1
                    continue
            sims = F.cosine_similarity(
                tgt[i].unsqueeze(0).expand(src.shape[0], -1), src)
            best_sim = sims.max()
            if best_sim > 0.5:
                loss = loss + (1.0 - best_sim)
            else:
                loss = loss + torch.clamp(best_sim - margin, min=0.0)
            n_pairs += 1

    if n_pairs == 0:
        return torch.tensor(0.0)

    return lambda_align * loss / n_pairs

def stat_based_init(model, num_data=None, cat_data=None, device="cpu"):
    """

    NumProj:       Linear(1→D) for each feature init throw FeatureStatEmbedder by statistics of each feature.
    CatEmbeddings: all rows Embedding init by one vector
                   by statistics ordinal-codes of this feature.
    """
    emb_dim = model.cls_token.shape[-1]
    stat_embedder = FeatureStatEmbedder(emb_dim=emb_dim).to(device)

    with torch.no_grad():
        if num_data is not None and model.num_proj is not None:
            stats = compute_feature_stats(num_data)  # [n_num, 8]
            stats_t = torch.tensor(stats, dtype=torch.float32).to(device)

            mean = stats_t.mean(dim=0, keepdim=True)
            std = stats_t.std(dim=0, keepdim=True) + 1e-8
            stats_t = (stats_t - mean) / std

            for i, proj_layer in enumerate(model.num_proj.proj):
                if i >= len(stats):
                    break
                emb = stat_embedder(stats_t[i].unsqueeze(0))  # [1, D]
                proj_layer.weight.data = emb.T  # [D, 1]
            print(f"[stat_init] NumProj inited "
                  f"({min(len(model.num_proj.proj), len(stats))} features all)")

        if cat_data is not None and len(model.cat_emb.embs) > 0:
            stats = compute_feature_stats(cat_data.astype(np.float32))  # [n_cat, 8]
            stats_t = torch.tensor(stats, dtype=torch.float32).to(device)

            mean = stats_t.mean(dim=0, keepdim=True)
            std = stats_t.std(dim=0, keepdim=True) + 1e-8
            stats_t = (stats_t - mean) / std

            for i, emb_layer in enumerate(model.cat_emb.embs):
                if i >= len(stats):
                    break
                emb = stat_embedder(stats_t[i].unsqueeze(0))  # [1, D]
                emb_layer.weight.data[:] = emb.squeeze(0)
            print(f"[stat_init] CatEmbeddings inited "
                  f"({min(len(model.cat_emb.embs), len(stats))} features all)")

def finetune_tabtransformer(
    cat_train, num_train, y_train,
    cat_val,   num_val,   y_val,
    cardinalities: list,
    TT: dict,
    backbone_checkpoint: str,
    freeze_mode: str = "last_layer",
    device: str = "cpu",
    matcher_a=None,
    num_features=None,
    cat_features=None,
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
                     if k.startswith("transformer.")
                     or k == "cls_token"
                     or k.startswith("input_norm.")
                     or k.startswith("pool_weights.")
                     }

    model.load_backbone_state_dict(backbone_keys)

    # Here is init of input layers based on statistics of features
    stat_based_init(model, num_data=num_train, cat_data=cat_train, device=device)

    with torch.no_grad():
        for proj in model.num_proj.proj:
            if torch.isnan(proj.weight).any():
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)

        for emb in model.cat_emb.embs:
            if torch.isnan(emb.weight).any():
                nn.init.normal_(emb.weight, mean=0.0, std=0.01)

    print("[init_check] Weights checked, NaN replaced with xavier/normal init")

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
    model_name = TT.get("model_name", f"tabtransformer_finetune_{freeze_mode}")
    save_path = os.path.join(save_dir, f"{model_name}.pth")

    source_embs = None
    lambda_align = TT.get("lambda_align", 0.05)
    print("Lambda align: ", lambda_align)

    emb_path = backbone_checkpoint.replace(".pth", "_feature_embs.pth")
    if os.path.exists(emb_path) and lambda_align > 0:
        try:
            source_embs = torch.load(emb_path, map_location=device, weights_only=False)
            print(f"[alignment_loss] Загружены эмбеддинги фичей Stage A: {emb_path}")
        except Exception as e:
            print(f"[alignment_loss] Не удалось загрузить эмбеддинги: {e}")

    match_indices = None
    if source_embs is not None and lambda_align > 0 and matcher_a is not None:

        skip_cols = {'Unnamed: 0'}
        num_features_clean = [c for c in (num_features or []) if c not in skip_cols]
        keep_idx = [i for i, c in enumerate(num_features or []) if c not in skip_cols]
        num_train_clean = num_train[:, keep_idx] if num_train is not None else num_train

        match_indices = matcher_a.get_match_indices_hungarian(
            num_train_clean,
            num_features_clean,
            cat_features or [],
            min_similarity=0.45
        )
        print(f"[alignment_loss] match_indices: "
              f"num={len(match_indices['num'])}, cat={len(match_indices['cat'])}")

    # if matcher_a is not None:
    #     col_names_target = (num_features or []) + (cat_features or [])
    #     num_for_match = num_train if num_train is not None else cat_train
    #     print("\n=== Feature Alignment (finetune) ===")
    #     matcher_a.match(num_for_match, col_names_target, verbose=True)
    train_start = time.time()

    for epoch in range(TT["epochs"]):
        model.train()
        total_epoch_loss = 0.0
        align_val = 0.0

        for batch in train_loader:
            cat = batch["cat"].to(device) if "cat" in batch else None
            num = batch["num"].to(device) if "num" in batch else None
            y = batch["y"].to(device)

            opt.zero_grad()

            out = model(cat, num)
            task_loss = loss_fn(out, y)

            # Alignment loss — sum with task_loss
            if source_embs is not None and lambda_align > 0:
                # target_embs = _extract_feature_embeddings(model, device)
                target_embs = _extract_feature_embeddings_grad(model, device)
                a_loss = alignment_loss(
                    source_embs, target_embs,
                    lambda_align=lambda_align,
                    match_indices=match_indices,  # mapping from matcher
                )
                total_loss = task_loss + a_loss
                align_val = a_loss.item()
            else:
                total_loss = task_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_epoch_loss += task_loss.item() * len(y)

        total_time = time.time() - train_start
        print(f"\n[finetune] Total training time: {total_time:.1f}s "
              f"({total_time / 60:.1f} min)")

        train_loss = total_epoch_loss / len(train_loader.dataset)

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

        align_str = f" | align={align_val:.4f}" if source_embs is not None else ""
        print(f"  Epoch {epoch + 1:3d}/{TT['epochs']} | "
              f"loss={train_loss:.4f}{align_str} | val_auc={auc:.4f}{marker}")

        if patience_counter >= patience:
            print(f"  Early stopping on epoch {epoch + 1}")
            break

    print(f"\n Best AUC={best_auc:.4f}")
    print(f"Checkpoint: {save_path}")

    # load the best wights to the model
    model.load_state_dict(torch.load(save_path, map_location=device))

    stage_embs = _extract_feature_embeddings(model, device)
    stage_emb_path = save_path.replace(".pth", "_feature_embs.pth")

    if source_embs is not None and match_indices is not None and matcher_a is not None:
        merged = merge_embeddings(
            source_embs_A=source_embs,
            source_embs_B=stage_embs,
            match_indices=match_indices,
            source_col_names_A=matcher_a.source_col_names,
            source_col_names_B=(num_features or []) + (cat_features or []),
            alpha=0.7
        )
        torch.save(merged, stage_emb_path)
        print(f"[merge_embeddings] Saved sum of embaddings: {stage_emb_path}")
        print(f"  Stage last features: {len(matcher_a.source_col_names)}")
        print(f"  Stage current features: {len((num_features or []) + (cat_features or []))}")
        print(f"  After merge:   {merged['num'].shape[0]} features")
    else:
        torch.save(stage_embs, stage_emb_path)
        print(f"[embeddings] Embedding were saved: {stage_emb_path}")

    return model, save_path


def adapt_to_new_dataset(
    cat_data, num_data, y_data,
    cardinalities: list,
    TT: dict,
    backbone_checkpoint: str,
    mode: str = "zero_shot",
    adapt_epochs: int = 5,
    device: str = "cpu",
    num_features: list = None,
    cat_features: list = None,
    matcher=None
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

    print(device)
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
    backbone_state = loaded.state_dict() if hasattr(loaded, 'state_dict') else loaded

    backbone_keys = {k: v for k, v in backbone_state.items()
                     if k.startswith("transformer.")
                     or k.startswith("input_norm.")
                     or k.startswith("pool_weights.")
                     or k == "cls_token"}

    model.load_backbone_state_dict(backbone_keys)

    #The same thing as in finetune - init input layers based on statistics from features
    stat_based_init(model, num_data=num_data, cat_data=cat_data, device=device)

    if mode == "zero_shot":
        # only inference
        model.freeze_backbone("full")
        print("  Zero-shot:")

    elif mode == "proj_adapt":
        # Backbone is frozen, learn only input+head
        freeze_mode = TT.get("freeze_mode", "full")
        model.freeze_backbone(freeze_mode)
        print("Current mode: ", freeze_mode)
        assert y_data is not None, "proj_adapt requires y_data (target for dataset)"

        # adapt_params = (
        #     list(model.cat_emb.parameters()) +
        #     (list(model.num_proj.parameters()) if model.num_proj else []) +
        #     list(model.cls.parameters())
        # )

        adapt_params = [p for p in model.parameters() if p.requires_grad]

        opt = torch.optim.Adam(adapt_params, lr=TT["lr"] * 0.3)
        # loss_fn = torch.nn.BCELoss()


        _weights = compute_class_weight("balanced", classes=np.unique(y_data), y=y_data)
        _w0 = torch.tensor(_weights[0], dtype=torch.float32).to(device)  # weight of good class 0
        _w1 = torch.tensor(_weights[1], dtype=torch.float32).to(device)  # weight of bad 1

        def loss_fn(pred, target):
            pred = torch.sigmoid(pred)
            weights = torch.where(target == 1, _w1, _w0)
            bce = torch.nn.functional.binary_cross_entropy(pred, target, reduction='none')
            return (bce * weights).mean()

        source_embs = None
        match_indices = None
        lambda_align = TT.get("lambda_align", 0.05)

        emb_path = backbone_checkpoint.replace(".pth", "_feature_embs.pth")
        matcher_path = backbone_checkpoint.replace(".pth", "_matcher.pkl")

        if os.path.exists(emb_path) and lambda_align > 0:
            try:
                source_embs = torch.load(emb_path, map_location=device, weights_only=False)
                print(f"[proj_adapt] Loaded embaddings: {emb_path}")
            except Exception as e:
                print(f"[proj_adapt] Failed with loading embaddings: {e}")

        if source_embs is not None and matcher is not None and num_features is not None:

            match_indices = matcher.get_match_indices_hungarian(
                num_data,
                num_features or [],
                cat_features or [],
                min_similarity=0.5
            )
            print(f"[proj_adapt] match_indices: "
                  f"num={len(match_indices['num'])}, "
                  f"cat={len(match_indices['cat'])}")

        adapt_ds = TabDataset(cat_data, num_data, y_data)
        adapt_loader = DataLoader(adapt_ds, batch_size=TT["batch_size"], shuffle=True)

        print(f"[alignment_loss] Using match_indices:")
        print(f"  num pairs: {len(match_indices['num'])}")
        print(f"  cat pairs: {len(match_indices['cat'])}")
        for i, j in list(match_indices['num'].items())[:5]:  # первые 5
            t = (num_features or [])[i] if i < len(num_features or []) else f"num[{i}]"
            s = matcher.source_col_names[j]
            print(f"    {t} → {s}")

        adapt_start = time.time()

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
                task_loss = loss_fn(out, y)

                if source_embs is not None and match_indices is not None and lambda_align > 0:
                    target_embs = _extract_feature_embeddings_grad(model, device)
                    a_loss = alignment_loss(
                        source_embs, target_embs,
                        lambda_align=lambda_align,
                        match_indices=match_indices,
                    )
                    total = task_loss + a_loss
                    align_val = a_loss.item()
                else:
                    total = task_loss

                total.backward()
                opt.step()
                total_loss += task_loss.item() * len(y)

            avg = total_loss / len(adapt_loader.dataset)
            align_str = f" | align={align_val:.4f}" if source_embs is not None else ""
            print(f"  Adapt epoch {epoch+1}/{adapt_epochs} | loss={avg:.4f}")

        adapt_time = time.time() - adapt_start
        print(f"\n[proj_adapt] Total adapt time: {adapt_time:.1f}s "
              f"({adapt_time / 60:.1f} min)")

    # Inference
    model.eval()
    ds = TabDataset(cat_data, num_data, y=None)
    loader = DataLoader(ds, batch_size=TT["batch_size"], shuffle=False)
    all_preds = []

    n_batches = len(loader)
    print(f"  Inference: {len(ds)} rows, {n_batches} batches (batch_size={TT['batch_size']})...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            cat = batch.get("cat", None)
            num = batch.get("num", None)
            if cat is not None:
                cat = cat.to(device)
            if num is not None:
                num = num.to(device)
            preds = model(cat, num)
            preds = torch.sigmoid(preds)
            all_preds.append(preds.cpu().numpy())
            if (i + 1) % max(1, n_batches // 5) == 0:
                print(f"  [{i + 1}/{n_batches} batch]")

    probas = np.concatenate(all_preds)
    return model, np.column_stack([1 - probas, probas])