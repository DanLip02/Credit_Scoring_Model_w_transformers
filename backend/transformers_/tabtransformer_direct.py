# tabtransformer_direct.py
import torch
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pickle
import tempfile


class FeatureSemanticMatcher:
    """
    fit(X_source, col_names)  — запомнить статистики и названия фичей датасета A
    match(X_target, col_names) — вернуть матрицу сходства [n_target, n_source]
    """

    def __init__(self, alpha: float = 0.4, beta: float = 0.6):
        # beta > alpha — distribution is more important than name
        self.alpha = alpha
        self.beta = beta
        self.source_stats = None
        self.source_col_names = None
        self._vectorizer = None
        self._source_name_embs = None

    #todo add BERT for semantic.
    def fit(self, X_source: np.ndarray, col_names: list) -> "FeatureSemanticMatcher":
        from .tab_transformers import compute_feature_stats

        self.source_col_names = list(col_names)
        self.source_stats = compute_feature_stats(X_source)

        names = [n.lower().replace("_", " ") for n in col_names]

        try:
            from sentence_transformers import SentenceTransformer
            print("[FeatureSemanticMatcher] BERT (multilingual)...")
            self._bert = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
            self._source_name_embs = self._bert.encode(
                names, show_progress_bar=False
            )  # [n_features, 384]
            self._use_bert = True
        except ImportError:
            from sklearn.feature_extraction.text import TfidfVectorizer
            print("[FeatureSemanticMatcher] sentence-transformers не установлен, "
                  "используем TF-IDF. Для лучшего качества: pip install sentence-transformers")
            self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            self._source_name_embs = self._vectorizer.fit_transform(names)
            self._use_bert = False

        return self

    def match(self, X_target: np.ndarray, target_col_names: list,
              verbose: bool = True) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity
        from .tab_transformers import compute_feature_stats

        assert self.source_stats is not None, "Error with fit"

        target_stats = compute_feature_stats(X_target)
        target_names = [n.lower().replace("_", " ") for n in target_col_names]

        # Семантическое сходство: BERT или TF-IDF
        if self._use_bert:
            target_embs = self._bert.encode(
                target_names, show_progress_bar=False
            )  # [n_target, 384]
            sem_sim = cosine_similarity(target_embs, self._source_name_embs)
        else:
            target_embs = self._vectorizer.transform(target_names)
            sem_sim = cosine_similarity(target_embs, self._source_name_embs)

        n_t, n_s = len(target_col_names), len(self.source_col_names)
        dist_sim = np.zeros((n_t, n_s), dtype=np.float32)
        for i in range(n_t):
            for j in range(n_s):
                t = target_stats[i];
                s = self.source_stats[j]
                t_n = (t - t.mean()) / (t.std() + 1e-8)
                s_n = (s - s.mean()) / (s.std() + 1e-8)
                dist_sim[i, j] = 1.0 / (1.0 + np.linalg.norm(t_n - s_n))

        similarity = self.alpha * sem_sim + self.beta * dist_sim

        if verbose:
            print("\n=== Feature Alignment ===")
            for i, t_name in enumerate(target_col_names):
                j = int(np.argmax(similarity[i]))
                flag = "✓" if similarity[i, j] > 0.6 else "~" if similarity[i, j] > 0.3 else "✗"
                print(f"  {flag} {t_name:25s} → {self.source_col_names[j]:25s} "
                      f"(total={similarity[i, j]:.3f}, "
                      f"sem={sem_sim[i, j]:.3f}, dist={dist_sim[i, j]:.3f})")
            print()

        return similarity.astype(np.float32)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        stats_norm = (stats - stats.mean(dim=-1, keepdim=True)) / (
                stats.std(dim=-1, keepdim=True) + 1e-8
        )
        return self.proj(stats_norm)

    def save(self, path: str):
        with open(path, "wb") as f: pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "FeatureSemanticMatcher":
        with open(path, "rb") as f: return pickle.load(f)


class DirectTabTransformer:
    """class for using custom TabTransformer without sklearn wrapper"""

    def __init__(self, **params):
        self.params = params
        self.model_ = None
        self.encoder_ = None
        self.scaler_ = None
        self.cat_features_ = params.get("cat_features")
        self.num_features_ = params.get("num_features")
        self.cardinalities_ = None
        self.device_ = params.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.transfer_mode_ = params.get("transfer_mode", None)
        self.backbone_path_ = params.get("backbone_path", None)
        self.freeze_mode_ = params.get("freeze_mode", "last_layer")  # "full"|"last_layer"|"none"
        self.adapt_epochs_ = params.get("adapt_epochs", 5),


    def prepare_data(self, X, y=None, X_val=None, y_val=None, fit=False):
        """preparing data for TabTransformer"""

        # Define features if not existed
        if self.cat_features_ is None:
            self.cat_features_ = list(X.select_dtypes(include=["object", "category"]).columns)
        else:
            self.cat_features_ = list(self.cat_features_)

        if self.num_features_ is None:
            self.num_features_ = list(X.select_dtypes(include=["int64", "float64"]).columns)
        else:
            self.num_features_ = list(self.num_features_)

        # Cardinalities (number of unique categories + 1 for padding)
        print("Cat categories in tabtransformer: ", self.cat_features_)
        if self.cat_features_ and (self.cardinalities_ is None or fit):
            self.cardinalities_ = [int(X[col].nunique()) + 1 for col in self.cat_features_]
            print(self.cardinalities_, type(self.cardinalities_))

        # Categorical features
        if fit or self.encoder_ is None:
            self.encoder_ = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            cat_data = self.encoder_.fit_transform(X[self.cat_features_].astype(str)) if self.cat_features_ else None
        else:
            cat_data = self.encoder_.transform(X[self.cat_features_].astype(str)) if self.cat_features_ else None

        if cat_data is not None:
            cat_data = np.clip(cat_data, 0, None)

        # Numerical features
        if self.num_features_:
            if fit or self.scaler_ is None:
                self.scaler_ = StandardScaler()
                num_data = self.scaler_.fit_transform(X[self.num_features_])
            else:
                num_data = self.scaler_.transform(X[self.num_features_])
        else:
            num_data = None

        # Prepare validation data if exists
        cat_val_data = num_val_data = None
        if X_val is not None:
            cat_val_data = self.encoder_.transform(
                X_val[self.cat_features_].astype(str)) if self.cat_features_ else None
            if self.num_features_:
                num_val_data = self.scaler_.transform(X_val[self.num_features_])

        return cat_data, num_data, cat_val_data, num_val_data

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from .train_transformers import fit_tabtransformer, finetune_tabtransformer, adapt_to_new_dataset

        cat_train, num_train, cat_val, num_val = self.prepare_data(
            X_train, y_train, X_val, y_val, fit=True
        )

        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split

            # Cleect only not None arrays
            arrays_to_split = [a for a in [cat_train, num_train] if a is not None]
            split_results = train_test_split(
                *arrays_to_split, y_train,
                test_size=0.2,
                stratify=y_train,
                random_state=42
            )

            # Unzip results
            # split_results = [arr_train, arr_val, ..., y_train_split, y_val_split]
            n = len(arrays_to_split)
            split_pairs = [(split_results[i * 2], split_results[i * 2 + 1]) for i in range(n)]
            y_train_split, y_val_split = split_results[-2], split_results[-1]

            idx = 0
            if cat_train is not None:
                cat_train, cat_val = split_pairs[idx]
                idx += 1
            else:
                cat_train, cat_val = None, None

            if num_train is not None:
                num_train, num_val = split_pairs[idx]
            else:
                num_train, num_val = None, None

            y_train = y_train_split
            y_val = y_val_split

        TT_config = {
            "embed_dim": self.params.get("embed_dim", 32),
            "n_heads": self.params.get("n_heads", 4),
            "n_layers": self.params.get("n_layers", 2),
            "mlp_dim": self.params.get("mlp_dim", 64),
            "dropout": self.params.get("dropout", 0.1),
            "lr": self.params.get("lr", 0.001),
            "batch_size": self.params.get("batch_size", 128),
            "epochs": self.params.get("epochs", 20),
            "class_weight": self.params.get("class_weight", "balanced"),
            "weight_decay": self.params.get("weight_decay", 0.01),
            "early_stopping_patience": self.params.get("early_stopping_patience", 10),
            "warmup_ratio": self.params.get("warmup_ratio", 0.1),
            "lambda_align": self.params.get("lambda_align", 0.0)
        }

        y_train_arr = y_train.values if hasattr(y_train, "values") else y_train
        y_val_arr = y_val.values if hasattr(y_val, "values") else y_val

        print("cardinalities: ", self.cardinalities_)

        if self.transfer_mode_ in (None, "pretrain", "finetune"):
            # Fit matcher on current dataset
            self.matcher_ = FeatureSemanticMatcher()
            num_for_match = num_train if num_train is not None else cat_train
            col_names = (self.num_features_ or []) + (self.cat_features_ or [])
            if num_for_match is not None and col_names:
                self.matcher_.fit(num_for_match, col_names)

                #todo check if finetune mode is activated because
                # it resave with new features and forget about backbone


                # if self.backbone_path_:
                #     matcher_path = self.backbone_path_.replace(".pth", "_matcher.pkl")
                # else:
                #     matcher_path = os.path.join(
                #         os.path.dirname(os.path.abspath(__file__)),
                #         "..", "..", "models", "transformers_", "matcher.pkl"
                #     )

                if self.backbone_path_:
                    if self.transfer_mode_ == "finetune":
                        matcher_path = self.backbone_path_.replace(".pth", "_stageB_matcher.pkl")
                    else:
                        matcher_path = self.backbone_path_.replace(".pth", "_matcher.pkl")
                else:
                    matcher_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "models", "transformers_", "matcher.pkl"
                    )

                self.matcher_.save(matcher_path)
                print(f"[Matcher] saved to: {matcher_path}")

        elif self.transfer_mode_ in ("zero_shot", "proj_adapt"):
            # load matcher из Stage A/B
            if self.backbone_path_:
                matcher_path = self.backbone_path_.replace(".pth", "_matcher.pkl")
                if os.path.exists(matcher_path):
                    self.matcher_ = FeatureSemanticMatcher.load(matcher_path)
                    num_for_match = num_train if num_train is not None else cat_train
                    col_names = (self.num_features_ or []) + (self.cat_features_ or [])
                    if num_for_match is not None and col_names:
                        self.matcher_.match(num_for_match, col_names, verbose=True)
                else:
                    print(f"[Matcher] File was not found: {matcher_path}")

        if self.transfer_mode_ in (None, "pretrain"):
            # Stage A — обычное обучение с нуля
            self.model_ = fit_tabtransformer(
                cat_train=cat_train, num_train=num_train,
                y_train=y_train_arr,
                cat_val=cat_val, num_val=num_val,
                y_val=y_val_arr,
                cardinalities=self.cardinalities_ or [],
                TT=TT_config,
                device=self.device_,
            )

        elif self.transfer_mode_ == "finetune":
            # Stage B — файнтюн с переносом backbone
            assert self.backbone_path_,  "transfer_mode='finetune' требует backbone_path в параметрах"
            matcher_path = self.backbone_path_.replace(".pth", "_matcher.pkl")

            if os.path.exists(matcher_path):
                matcher_a = FeatureSemanticMatcher.load(matcher_path)

                print(f"[Matcher] source_cols ({len(matcher_a.source_col_names)}): {matcher_a.source_col_names}")
                print(f"[Matcher] source_stats shape: {matcher_a.source_stats.shape}")

                col_names_b = (self.num_features_ or []) + (self.cat_features_ or [])
                num_for_match = num_train if num_train is not None else cat_train
                print("\n=== Feature mapping: Stage A → Stage B ===")
                matcher_a.match(num_for_match, col_names_b, verbose=True)

            self.model_, _ = finetune_tabtransformer(
                cat_train=cat_train, num_train=num_train,
                y_train=y_train_arr,
                cat_val=cat_val, num_val=num_val,
                y_val=y_val_arr,
                cardinalities=self.cardinalities_ or [],
                TT=TT_config,
                backbone_checkpoint=self.backbone_path_,
                freeze_mode=self.freeze_mode_,
                device=self.device_,
            )

        elif self.transfer_mode_ in ("zero_shot", "proj_adapt"):
            # Stage C — адаптация к новому датасету
            assert self.backbone_path_, "transfer_mode='zero_shot'/'proj_adapt' требует backbone_path"
            self.model_, _ = adapt_to_new_dataset(
                cat_data=cat_train, num_data=num_train,
                y_data=y_train_arr,
                cardinalities=self.cardinalities_ or [],
                TT=TT_config,
                backbone_checkpoint=self.backbone_path_,
                mode=self.transfer_mode_,
                adapt_epochs=self.adapt_epochs_,
                device=self.device_,
            )

        else:
            raise ValueError(
                f"Неизвестный transfer_mode: {self.transfer_mode_!r}. "
                f"Допустимые: None, 'pretrain', 'finetune', 'zero_shot', 'proj_adapt'"
            )

        return self

        # self.model_ = fit_tabtransformer(
        #     cat_train=cat_train,
        #     num_train=num_train,
        #     y_train=y_train.values if hasattr(y_train, 'values') else y_train,
        #     cat_val=cat_val,
        #     num_val=num_val,
        #     y_val=y_val.values if hasattr(y_val, 'values') else y_val,
        #     cardinalities=self.cardinalities_ or [],
        #     TT=TT_config,
        #     device=self.device_
        # )

        # return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fitted")

        cat_data, num_data, _, _ = self.prepare_data(X, fit=False)

        from .train_transformers import TabDataset
        from torch.utils.data import DataLoader

        dataset = TabDataset(cat_data, num_data)
        loader = DataLoader(
            dataset,
            batch_size=self.params.get("batch_size", 128),
            shuffle=False
        )

        self.model_.eval()
        all_preds = []

        n_batches = len(loader)
        print(f"[predict_proba] {len(dataset)} rows, {n_batches} batch...")

        with torch.no_grad():
            for i, batch in enumerate(loader):

                cat = batch["cat"].to(self.device_) if "cat" in batch else None
                num = batch["num"].to(self.device_) if "num" in batch else None
                preds = self.model_(cat, num)
                all_preds.append(preds.cpu().numpy())

                if (i + 1) % max(1, n_batches // 5) == 0:
                    print(f"[predict_proba] {i + 1}/{n_batches} batch")

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