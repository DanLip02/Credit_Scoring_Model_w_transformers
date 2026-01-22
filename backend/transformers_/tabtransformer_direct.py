# tabtransformer_direct.py
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pickle
import tempfile


class DirectTabTransformer:
    """class for using custom TabTransformer without sklearn wrapper"""

    def __init__(self, **params):
        self.params = params
        self.model_ = None
        self.encoder_ = None
        self.scaler_ = None
        self.cat_features_ = None
        self.num_features_ = None
        self.cardinalities_ = None
        self.device_ = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def prepare_data(self, X, y=None, X_val=None, y_val=None, fit=False):
        """preparing data for TabTransformer"""

        print(X.dtypes)
        print(self.num_features_)

        # Define features if not existed
        if self.cat_features_ is None:
            self.cat_features_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if self.num_features_ is None:
            self.num_features_ = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        # categorial features
        if fit or self.encoder_ is None:
            self.encoder_ = OrdinalEncoder()
            cat_data = self.encoder_.fit_transform(X[self.cat_features_].astype(str))
        else:
            cat_data = self.encoder_.transform(X[self.cat_features_].astype(str))

        # numerical features
        if len(self.num_features_) > 0:
            if fit or self.scaler_ is None:
                self.scaler_ = StandardScaler()
                num_data = self.scaler_.fit_transform(X[self.num_features_])
            else:
                num_data = self.scaler_.transform(X[self.num_features_])
        else:
            num_data = None

        # cardinalitons (only using with fit)
        if fit and self.cat_features_:
            self.cardinalities_ = [int(X[col].nunique()) + 1 for col in self.cat_features_]

        # prepare validation data if not existed
        cat_val_data = None
        num_val_data = None

        if X_val is not None:
            cat_val_data = self.encoder_.transform(X_val[self.cat_features_].astype(str))
            if self.scaler_ is not None and len(self.num_features_) > 0:
                num_val_data = self.scaler_.transform(X_val[self.num_features_])

        return cat_data, num_data, cat_val_data, num_val_data

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """learning model on prepared train/val data"""
        from .train_transformers import fit_tabtransformer

        # prepare data
        cat_train, num_train, cat_val, num_val = self.prepare_data(
            X_train, y_train, X_val, y_val, fit=True
        )
        print(cat_train, num_train, cat_val, num_val)
        # using existed X_val/y_val or create from X_train/y_train
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            cat_train, cat_val, num_train, num_val, y_train_split, y_val_split = train_test_split(
                cat_train, num_train, y_train,
                test_size=0.2,
                stratify=y_train,
                random_state=42
            )
            y_val = y_val_split

        # configuration
        TT_config = {
            "embed_dim": self.params.get("embed_dim", 32),
            "n_heads": self.params.get("n_heads", 4),
            "n_layers": self.params.get("n_layers", 2),
            "mlp_dim": self.params.get("mlp_dim", 64),
            "dropout": self.params.get("dropout", 0.1),
            "lr": self.params.get("lr", 0.001),
            "batch_size": self.params.get("batch_size", 128),
            "epochs": self.params.get("epochs", 20)
        }

        # Learn model
        self.model_ = fit_tabtransformer(
            cat_train=cat_train,
            num_train=num_train,
            y_train=y_train.values if hasattr(y_train, 'values') else y_train,
            cat_val=cat_val,
            num_val=num_val,
            y_val=y_val.values if hasattr(y_val, 'values') else y_val,
            cardinalities=self.cardinalities_,
            TT=TT_config,
            device=self.device_
        )

        return self

    def predict_proba(self, X):
        """predict probabilities"""
        if self.model_ is None:
            raise RuntimeError("Model not fitted")

        # preparing data
        cat_data, num_data, _, _ = self.prepare_data(X, fit=False)

        # Create Dataset & DataLoader
        from .train_transformers import TabDataset
        from torch.utils.data import DataLoader

        dataset = TabDataset(cat_data, num_data)
        loader = DataLoader(
            dataset,
            batch_size=self.params.get("batch_size", 128),
            shuffle=False
        )

        # prediction
        self.model_.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                cat = batch["cat"].to(self.device_)
                num = batch["num"].to(self.device_) if "num" in batch else None
                preds = self.model_(cat, num)
                all_preds.append(preds.cpu().numpy())

        probas = np.concatenate(all_preds)
        return np.column_stack([1 - probas, probas])

    def predict(self, X):
        """predicted classes"""
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)

    def save_preprocessors(self, path):
        """save preprocessing"""
        with open(path, 'wb') as f:
            pickle.dump({
                'encoder': self.encoder_,
                'scaler': self.scaler_,
                'cat_features': self.cat_features_,
                'num_features': self.num_features_,
                'cardinalities': self.cardinalities_
            }, f)

    def load_preprocessors(self, path):
        """loading preprocessing"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.encoder_ = data['encoder']
            self.scaler_ = data['scaler']
            self.cat_features_ = data['cat_features']
            self.num_features_ = data['num_features']
            self.cardinalities_ = data['cardinalities']

    def get_params(self, deep=True):
        return self.params.copy()

    def set_params(self, **params):
        self.params.update(params)
        return self