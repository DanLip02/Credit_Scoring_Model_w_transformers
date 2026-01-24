import pandas as pd
import kagglehub
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
from pathlib import Path
import yaml

def load_german_credit_risk():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Поднимаемся из backend в корень
    file_path = os.path.join(project_root, 'data', 'german_credit_data.csv')

    print(f"Searching file by path : {file_path}")
    df = pd.read_csv(file_path)

    return df

def load_data_old(filepath="german_credit_data.csv"):
    df = pd.read_csv(filepath)
    df["target"] = df["target"].map({1:0, 2:1})
    return df

def load_config(cfg_path: str):
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def clean_target(df: pd.DataFrame, target_col: str, maps_num: dict) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"target  '{target_col}' was not found ")

    before = len(df)
    df = df.dropna(subset=[target_col])
    if len(df) < before:
        print(f"⚠️ Deleted {before - len(df)} rows with NaN in target '{target_col}'")

    # df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    # df[target_col] = df[target_col].map(maps_num) if target_col in df.columns and maps_num is not None else df[target_col]
    # unknown_mask = df[target_col].isna()
    # if unknown_mask.any():
    #     print(f"⚠️ Unknown classes: {df.loc[unknown_mask, target_col].unique()}")
    #     df = df.loc[~unknown_mask]

    return df

def prepare_features(df: pd.DataFrame, cfg: dict):
    #todo get from validation api

    features_cfg = cfg["features"]

    num_features = features_cfg.get("numeric", None)
    cat_features = features_cfg.get("categorical", None)
    target_col = cfg["data"].get("target")
    maps_num = cfg["data"].get("mapper", None)
    skip = cfg["data"].get("skip", None)
    date_col = cfg["data"].get("date_column", None)
    filter_col = cfg["data"].get("filter_columns", None)
    df = df[df[skip] == 0] if skip is not None else df

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values(filter_col) if filter_col is not None else df

    print(df.columns)

    df = clean_target(df, target_col, maps_num)

    if num_features is not None and cat_features is not None:
        X = df[num_features + cat_features].copy()

        for col in num_features:
            X[col] = X[col].fillna(0)

        for col in cat_features:
            X[col] = X[col].fillna("missing").astype(str)

    else:
        X = df.drop(columns=[target_col])

    y = df[target_col].map(maps_num) if target_col in df.columns and maps_num is not None else df[target_col]
    #todo add check for nums (linear mapping)
    # y = df[target_col] if target_col in df.columns else None

    cat_features = X.select_dtypes(include=["object"]).columns
    num_features = X.select_dtypes(include=["int64", "float64"]).columns

    X = X.reset_index(drop=True)

    return X, y.reset_index(drop=True), num_features, cat_features

def apply_data(type_data: str, data: dict=None):

    if data:
        cfg_data = data
    else:
        cfg_data = load_config(f"data_yaml/data_{type_data}.yaml")

    # cfg_split = load_config("configs/train_split.yaml")

    df = load_data(cfg_data)
    print(f"✅ Uploaded {len(df)} rows and  {df.shape[1]} columns.")

    X, y, num_features, cat_features = prepare_features(df, cfg_data)


    print(f"📊 Numerical features: {num_features}")
    print(f"🏷️ Categorical features: {cat_features}")
    if y is not None:
        print(f"🎯 Target: {cfg_data['data']['target']}")

    # X_train, X_test, y_train, y_test = split_data(X, y, cfg_split["split"])
    # print(f"🔹 Train: {X_train.shape}, Test: {X_test.shape}")

    return X, y, num_features, cat_features


def load_data(cfg: dict):
    # import pyreadstat
    """
        load data based on  cfg["data"].
        supported formats: Excel, CSV, TSV, SAV (SPSS), JSON, Parquet.
        """

    data_cfg = cfg["data"]
    path = data_cfg["path"]

    sheet_name = data_cfg.get("sheet_name", None)
    skiprows = data_cfg.get("skiprows", None)
    dropna_flag = data_cfg.get("dropna", False)

    file_format = data_cfg.get("format", None)
    if file_format is None:
        file_format = path.split(".")[-1].lower()

    if file_format in ("xlsx", "xls"):
        df = pd.read_excel(path, sheet_name=sheet_name, skiprows=skiprows)

    elif file_format == "csv":
        df = pd.read_csv(path, skiprows=skiprows)

    elif file_format == "tsv":
        df = pd.read_csv(path, sep="\t", skiprows=skiprows)

    # elif file_format == "sav":
    #     df, meta = pyreadstat.read_sav(path)
    #
    # elif file_format == "sas7bdat":
    #     df, meta = pyreadstat.read_sas7bdat(path)

    elif file_format == "json":
        df = pd.read_json(path)

    elif file_format == "parquet":
        df = pd.read_parquet(path)

    else:
        raise ValueError(f"file format '{file_format}' is not supported")

    if dropna_flag:
        df = df.dropna()

    return df

def add_features(df):
    # example: add log credit amount and amount per month
    # if "credit_amount" in df.columns and "duration_month" in df.columns:
    df["credit_amount_log"] = np.log1p(df["credit_amount"])
    df["amount_per_month"] = df["credit_amount"] / (df["duration_month"].replace(0,1))
    return df

def split_data(target_col: pd.Series,
               df: pd.DataFrame,
               cfg_split: dict,
               method: str,
               custom_col: str = None):
    """
    Function to split data into X_train, X_test, y_train, y_test
    based on type of split from yaml config.

    Args:
        target_col (pd.Series): target y
        df (pd.DataFrame): matrices of featured X
        cfg_split (dict): config with structure {'type': ..., 'params': {...}}
        method (str): type split ('base', 'time', 'kfold', 'custom')
        custom_col (str, optional): custom split additional column (ex, id)

    Returns:
        X_train, X_test, y_train, y_test
    """
    params = cfg_split.get("params", {})

    # checks
    # if not isinstance(df, pd.DataFrame):
    #     raise TypeError("❌ ARG df must be pandas.DataFrame")
    # if not isinstance(target_col, (pd.Series, pd.DataFrame)):
    #     raise TypeError("❌ ARG target_col must be pandas.Series or DataFrame")
    if len(df) != len(target_col):
        raise ValueError(f"❌ Shape X ({len(df)}) and y ({len(target_col)}) not same ")

    if method == "base":
        stratify_vals = target_col if params.get("stratify", False) else None
        X_train, X_test, y_train, y_test = train_test_split(
            df,
            target_col,
            test_size=params.get("test_size", 0.2),
            random_state=params.get("random_state", 42),
            stratify=stratify_vals
        )
        print("trouble after split")
        assert isinstance(X_train, pd.DataFrame), "X_train must be DataFrame"
        assert isinstance(X_test, pd.DataFrame), "X_test must be DataFrame"

    elif method == "time":
        date_col = params["date_column"]
        split_date = pd.to_datetime(params["split_date"])

        mask_train = df[date_col] <= split_date
        mask_test = df[date_col] > split_date

        X_train, X_test = df.loc[mask_train], df.loc[mask_test]
        y_train, y_test = target_col.loc[mask_train], target_col.loc[mask_test]

    elif method == "kfold":
        kf = KFold(
            n_splits=params.get("n_splits", 5),
            shuffle=params.get("shuffle", True),
            random_state=params.get("random_state", 42)
        )
        return list(kf.split(df, target_col))

    elif method == "custom":
        id_col = custom_col if custom_col is not None else params.get("id_column", "global_id_ogrn")
        if id_col not in df.columns:
            raise ValueError(f"❌ Column '{id_col}' was not found in  X")

        unique_ids = df[id_col].unique()
        train_ids, test_ids = train_test_split(
            unique_ids,
            test_size=params.get("test_size", 0.2),
            random_state=params.get("random_state", 42),
            shuffle=params.get("shuffle", True)
        )

        train_mask = df[id_col].isin(train_ids)
        test_mask = df[id_col].isin(test_ids)

        X_train, X_test = df.loc[train_mask], df.loc[test_mask]
        y_train, y_test = target_col.loc[train_mask], target_col.loc[test_mask]

    else:
        raise ValueError(f"❌ Type split '{method}' is not supported")

    print(f"✅ Split done: {method}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test

def split_data_old(df, test_size=0.2, random_state=42):
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