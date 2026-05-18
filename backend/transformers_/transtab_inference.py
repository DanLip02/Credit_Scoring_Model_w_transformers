import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, roc_curve
import transtab
import time
import warnings
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

warnings.filterwarnings('ignore')

np.Inf = np.inf
np.NaN = np.nan


def print_metrics(y_true, y_pred_proba, threshold, label=""):
    y_pred = (y_pred_proba > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr_val = tp / (tp + fn + 1e-8)
    fnr = fn / (tp + fn + 1e-8)
    fpr_val = fp / (fp + tn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision * tpr_val / (precision + tpr_val + 1e-8)
    auc = roc_auc_score(y_true, y_pred_proba)

    print(f"\n{'=' * 50}")
    print(f"МЕТРИКИ TransTab ({label}, порог={threshold:.4f}):")
    print(f"{'=' * 50}")
    print(f"AUC:       {auc:.4f}")
    # print(f"KS:        {ks:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"TPR:       {tpr_val:.4f}")
    print(f"FPR:       {fpr_val:.4f}")
    print(f"TNR:       {tnr:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Матрица:   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    return auc

def prepare_dataset(path, target_col, drop_cols):
    df = pd.read_csv(path)
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.dropna(subset=[target_col])
    df[target_col] = df[target_col].astype(float)

    cat_cols, num_cols, bin_cols = [], [], []
    X = df.drop(columns=[target_col])
    for col in X.columns:
        if X[col].dtype == 'object':
            cat_cols.append(col)
        elif X[col].nunique() == 2:
            bin_cols.append(col)
        else:
            num_cols.append(col)

    for col in bin_cols:
        vals = sorted(df[col].dropna().unique())
        mapping = {vals[0]: 0, vals[1]: 1}
        df[col] = df[col].map(mapping).astype(int)

    for col in cat_cols:
        df[col] = df[col].astype(str)

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)

    print(f"  shape={df.shape}, cat={len(cat_cols)}, num={len(num_cols)}, bin={len(bin_cols)}")
    return df, target_col, cat_cols, num_cols, bin_cols

def main():

    path = r""
    df_C, target_C, cat_C, num_C, bin_C = prepare_dataset(
        path= path,
        target_col="target",
        drop_cols=["Customer_ID"]
    )

    X_C = df_C.drop(columns=[target_C])
    y_C = df_C[target_C]
    X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(
        X_C, y_C, test_size=0.2, random_state=42, stratify=y_C
    )

    print(f"\nTrain: {X_train_C.shape}, Test: {X_test_C.shape}")
    print(f"Target rate: train={y_train_C.mean():.4f}, test={y_test_C.mean():.4f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model = transtab.build_classifier(
        categorical_columns=cat_C,
        numerical_columns=num_C,
        binary_columns=bin_C,
    )
    model = model.to(device)

    print("=" * 55)

    trainset_C = (X_train_C, y_train_C)
    valset = (X_test_C, y_test_C)

    start = time.time()
    transtab.train(
        model,
        trainset_C,
        valset,
        num_epoch=25,
        batch_size=128,
        lr=1e-4,
        eval_metric='auc',
        patience=5,
        num_workers=0,
        output_dir='./transtab_checkpoints',
    )
    train_time = time.time() - start
    print(f"\nlearning: {train_time:.1f} сек ({train_time / 60:.1f} мин)")

    fake_y = pd.Series(np.zeros(len(X_test_C)), dtype=float)

    start = time.time()
    y_pred_proba = transtab.predict(
        model, X_test_C,
        y_test=fake_y,
        return_loss=False,
        eval_batch_size=256,
    )
    predict_time = time.time() - start

    print(f"\nPredict: {predict_time:.2f} сек")
    print(f"y_prob: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")

    fpr, tpr, thresholds = roc_curve(y_test_C, y_pred_proba)
    auc_score = roc_auc_score(y_test_C, y_pred_proba)
    ks = float(np.max(tpr - fpr))

    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': auc_score
    }

    precisions, recalls, pr_thresholds = precision_recall_curve(y_test_C, y_pred_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_threshold = pr_thresholds[np.argmax(f1_scores[:-1])]
    print(f"\nOptimal threshold (по F1): {best_threshold:.4f}")

    auc_05 = print_metrics(y_test_C, y_pred_proba, threshold=0.5, label="порог 0.5")
    auc_best = print_metrics(y_test_C, y_pred_proba, threshold=best_threshold, label="опт. порог")


    print(f"{'=' * 50}")
    print(f"TabTransformer (transfer learning, 5 datasets):  0.8916")
    print(f"TransTab (no pretrain, only dataset C):         {auc_score:.4f}")
    print(f"diff:  {0.8916 - auc_score:+.4f}")

    return {
        'model': model,
        'roc_data': roc_data,
        'y_test': y_test_C,
        'y_pred_proba': y_pred_proba,
        'auc': auc_score,
        'ks': ks
    }


if __name__ == '__main__':

    print(dir(transtab))
    # result = main()