import pandas as pd
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np

def load_german_credit_risk():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Поднимаемся из backend в корень
    file_path = os.path.join(project_root, 'data', 'german_credit_data.csv')

    print(f"Searching file by path : {file_path}")
    df = pd.read_csv(file_path)

    return df

def load_data(filepath="german_credit_data.csv"):
    df = pd.read_csv(filepath)
    df["target"] = df["target"].map({1:0, 2:1})
    return df

def add_features(df):
    # example: add log credit amount and amount per month
    # if "credit_amount" in df.columns and "duration_month" in df.columns:
    df["credit_amount_log"] = np.log1p(df["credit_amount"])
    df["amount_per_month"] = df["credit_amount"] / (df["duration_month"].replace(0,1))
    return df

def split_data(df, test_size=0.2, random_state=42):
    X = df.drop(["target", "credit_amount", 'Unnamed: 0'], axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def get_preprocessor(X_train):
    categorical = X_train.select_dtypes(include=["object"]).columns
    numeric = X_train.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        ]
    )
    return preprocessor


if __name__ == '__main__':
    pass