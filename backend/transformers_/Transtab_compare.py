import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix, roc_curve
import transtab
import time
import warnings
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.memory_allocated() / 1e9, "GB")

warnings.filterwarnings('ignore')

def main():
    warnings.filterwarnings('ignore')


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


    print("Dataset A (HomeCredit):")
    df_A, target_A, cat_A, num_A, bin_A = prepare_dataset(
        path=r"C:\Users\Danch\PycharmProjects\Credit_Scoring_Model_w_transformers\data\Dataset_A\application_train_all_models_final_2.csv",
        target_col="TARGET",
        drop_cols=["SK_ID_CURR", "Unnamed: 0"]
    )

    print("Dataset B (GiveMeCredit):")
    df_B, target_B, cat_B, num_B, bin_B = prepare_dataset(
        path=r"C:\Users\Danch\PycharmProjects\Credit_Scoring_Model_w_transformers\data\Dataset_B\train_baseline_ML.csv",
        target_col="SeriousDlqin2yrs",
        drop_cols=["Unnamed: 0"]
    )

    print("Dataset C (CreditScoring):")
    df_C, target_C, cat_C, num_C, bin_C = prepare_dataset(
        path=r"C:\Users\Danch\PycharmProjects\Credit_Scoring_Model_w_transformers\data\Dataset_C\credit_scoring_full_raw.csv",
        target_col="target",
        drop_cols=["Customer_ID"]
    )

    X_C = df_C.drop(columns=[target_C])
    y_C = df_C[target_C]
    X_train_C, X_test_C, y_train_C, y_test_C = train_test_split(
        X_C, y_C, test_size=0.2, random_state=42, stratify=y_C
    )
    X_train_C = X_train_C.reset_index(drop=True)
    X_test_C = X_test_C.reset_index(drop=True)
    y_train_C = y_train_C.reset_index(drop=True)
    y_test_C = y_test_C.reset_index(drop=True)


    X_A = df_A.drop(columns=[target_A]).reset_index(drop=True)
    y_A = df_A[target_A].reset_index(drop=True)


    X_B = df_B.drop(columns=[target_B]).reset_index(drop=True)
    y_B = df_B[target_B].reset_index(drop=True)

    print(f"\nDataset A train: {X_A.shape}")
    print(f"Dataset B train: {X_B.shape}")
    print(f"Dataset C train: {X_train_C.shape}, test: {X_test_C.shape}")


    all_cat = list(set(cat_A + cat_B + cat_C))
    all_num = list(set(num_A + num_B + num_C))
    all_bin = list(set(bin_A + bin_B + bin_C))


    all_bin = list(set(all_bin))
    all_cat = [c for c in all_cat if c not in all_bin]
    all_num = [c for c in all_num if c not in all_bin and c not in all_cat]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    model = transtab.build_classifier(
        categorical_columns=all_cat,
        numerical_columns=all_num,
        binary_columns=all_bin,
    )
    model = model.to(device)


    print("=" * 55)

    trainset_multi = [
        (X_A, y_A),
        (X_B, y_B),
        (X_train_C, y_train_C),
    ]
    valset = (X_test_C, y_test_C)

    start = time.time()
    transtab.train(
        model,
        trainset_multi,
        valset,
        num_epoch=20,
        batch_size=128,
        lr=1e-4,
        eval_metric='auc',
        patience=5,
        num_workers=0,
        output_dir='./transtab_pretrain',
    )
    print(f"Pretrain: {time.time() - start:.1f} сек")

    print("=" * 55)

    trainset_C = (X_train_C, y_train_C)

    start = time.time()
    transtab.train(
        model,
        trainset_C,
        valset,
        num_epoch=30,
        batch_size=64,
        lr=0.0005,
        eval_metric='auc',
        patience=10,
        num_workers=0,
        output_dir='./transtab_finetune',
    )
    print(f"Finetune: {time.time() - start:.1f} сек")

    fake_y = pd.Series(np.zeros(len(X_test_C)), dtype=float)
    y_pred_proba = transtab.predict(
        model, X_test_C,
        y_test=fake_y,
        return_loss=False,
        eval_batch_size=256,
    )
    print(f"\nPredict: min={y_pred_proba.min():.4f}, max={y_pred_proba.max():.4f}, mean={y_pred_proba.mean():.4f}")

    precisions, recalls, thresholds = precision_recall_curve(y_test_C, y_pred_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores[:-1])]
    print(f"optimal threshold: {best_threshold:.4f}")

    def print_full_metrics(y_true, y_pred_proba, threshold, label=""):
        y_pred = (y_pred_proba > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn + 1e-8)
        fnr = fn / (tp + fn + 1e-8)
        fpr_val = fp / (fp + tn + 1e-8)
        tnr = tn / (tn + fp + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        acc = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * precision * tpr / (precision + tpr + 1e-8)
        auc = roc_auc_score(y_true, y_pred_proba)
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_pred_proba)
        ks = float(np.max(tpr_curve - fpr_curve))

        print(f"{'=' * 50}")
        print(f"AUC:       {auc:.4f}")
        print(f"KS:        {ks:.4f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"TPR:       {tpr:.4f}  (Sensitivity)")
        print(f"FNR:       {fnr:.4f}  (Miss Rate)")
        print(f"FPR:       {fpr_val:.4f}  (Fall-out)")
        print(f"TNR:       {tnr:.4f}  (Specificity)")
        print(f"F1:        {f1:.4f}")
        print(f"{'─' * 50}")
        print(f"  TN={tn:6d}  FP={fp:6d}")
        print(f"  FN={fn:6d}  TP={tp:6d}")

    print_full_metrics(y_test_C, y_pred_proba, threshold=0.5, label="порог 0.5")
    print_full_metrics(y_test_C, y_pred_proba, threshold=best_threshold, label="опт. порог")

    auc_final = roc_auc_score(y_test_C, y_pred_proba)

    print(f"{'=' * 50}")
    print(f"TabTransformer (transfer learning):      0.8916")
    print(f"TransTab (without pretrain):                 0.6778")
    print(f"TransTab (with pretrain A+B+C → finetune):  {auc_final:.4f}")


if __name__ == '__main__':
    main()