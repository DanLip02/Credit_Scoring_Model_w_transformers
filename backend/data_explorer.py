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

def split_data(df, test_size=0.2, random_state=42):
    X = df.drop("target", axis=1)
    log_credit_amount = np.log1p(X["credit_amount"])
    X = df.drop("credit_amount", axis=1)
    # добавляем новый столбец в X
    X = pd.concat([X, log_credit_amount.rename("credit_amount_log")], axis=1)
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