# from data_explorer.data_explorer import *
from .train_transformers import fit_tabtransformer
import torch
import os

def run_transformers(X_train, y_train, X_test, y_test, config_param):
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # ====== transform data ======
    # OneHotEncoder gives sparse output, лучше использовать OrdinalEncoder для TabTransformer
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    enc = OrdinalEncoder()
    cat_train = enc.fit_transform(X_train[cat_cols].astype(str))
    cat_val = enc.transform(X_test[cat_cols].astype(str))

    if len(num_cols) > 0:
        scaler = StandardScaler()
        num_train = scaler.fit_transform(X_train[num_cols])
        num_val = scaler.transform(X_test[num_cols])
    else:
        num_train = num_val = None

    # ====== cardinalities ======
    cardinalities = [int(X_train[c].nunique()) for c in cat_cols]

    # ====== device ======
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ====== create folders ======
    try:
        os.makedirs(r"models\transformers", exist_ok=True)
    except:
        print("can not be created dir")
    # ====== fit TabTransformer ======
    model = fit_tabtransformer(
        cat_train, num_train, y_train.values,
        cat_val, num_val, y_test.values,
        cardinalities, config_param,
        device=device
    )

    print("Training finished. Best model saved to models/transformers/tabtransformer_best.pth")