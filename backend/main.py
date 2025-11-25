from data_explorer import *
from baselines import run_model, get_baseline_models
from train_transformers import fit_tabtransformer
import numpy as np
import torch
import os



if __name__ == '__main__':
    # df = load_data("/Users/danilalipatov/Credit_Scoring_Model_w_transformers/backend/datasets/german_credit_risk/german_credit_risk.csv")
    df = load_data(r"C:\Users\Danch\PycharmProjects\Credit_Scoring_Model_w_transformers\backend\datasets\german_credit_risk\german_credit_risk.csv")
    df = add_features(df)

    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = get_preprocessor(X_train)

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
    os.makedirs(r"C:\Users\Danch\PycharmProjects\Credit_Scoring_Model_w_transformers\models\transformers", exist_ok=True)

    # ====== fit TabTransformer ======
    model = fit_tabtransformer(
        cat_train, num_train, y_train.values,
        cat_val, num_val, y_test.values,
        cardinalities,
        device=device
    )

    print("Training finished. Best model saved to models/transformers/tabtransformer_best.pth")

    # models = get_baseline_models()
    # results = {}
    #
    # for name, model in models.items():
    #     results[name] = run_model(preprocessor, X_train, X_test, y_train, y_test, model, name)
