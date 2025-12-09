# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
# from config import RANDOM_SEED, TEST_SIZE

def add_features(df):
    # example: add log credit amount and amount per month
    if "credit_amount" in df.columns and "duration_month" in df.columns:
        df["credit_amount_log"] = np.log1p(df["credit_amount"])
        df["amount_per_month"] = df["credit_amount"] / (df["duration_month"].replace(0,1))
    return df

def split_df(df, target="target", test_size=None , random_state=42, stratify=True):
    X = df.drop(columns=[target])
    y = df[target]
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)

def get_feature_lists(X):
    numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    # remove index-like numeric if any
    return numeric, categorical

def build_preprocessor(numeric, categorical):
    # numeric: scale, categorical: ordinal (for transformer embeddings we will map later)
    num_pipe = ("num", StandardScaler(), numeric) if numeric else ("num", "passthrough", [])
    cat_pipe = ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical) if categorical else ("cat", "passthrough", [])
    ct = ColumnTransformer(transformers=[num_pipe, cat_pipe], remainder="drop")
    return ct

# helper to apply SMOTE on already-transformed numpy arrays
def apply_smote(X_trans, y, RANDOM_SEED=42):
    sm = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = sm.fit_resample(X_trans, y)
    return X_res, y_res
