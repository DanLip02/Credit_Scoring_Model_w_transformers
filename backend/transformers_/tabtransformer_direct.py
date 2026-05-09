# tabtransformer_direct.py
import torch
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import pickle
from scipy.optimize import linear_sum_assignment
import tempfile
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .tab_transformers import compute_feature_stats, _normalize_for_stats
import time

class FeatureSemanticMatcher:
    """
    fit(X_source, col_names)  — запомнить статистики и названия фичей датасета A
    match(X_target, col_names) — вернуть матрицу сходства [n_target, n_source]
    """

    def __init__(self, alpha: float = 0.8, beta: float = 0.2, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        # beta > alpha — distribution is more important than name
        self.alpha = alpha
        self.beta = beta
        self.source_stats = None
        self.source_col_names = None
        self._vectorizer = None
        self._source_name_embs = None
        self._bert = SentenceTransformer(model_name)

    #todo add BERT for semantic.
    def fit(self, X_source: np.ndarray, col_names: list, dataset_name=None) -> "FeatureSemanticMatcher":

        self.source_col_names = list(col_names)
        self.source_stats = compute_feature_stats(_normalize_for_stats(X_source))
        self.source_dataset_name = dataset_name

        print("[fit] source_stats (skew, kurt) для первых 5 фичей:")
        for i, name in enumerate(col_names[:5]):
            print(f"  {name}: skew={self.source_stats[i][2]:.3f}, kurt={self.source_stats[i][3]:.3f}")

        names = self.get_enriched_names(col_names, dataset_name)

        try:
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
            print("[FeatureSemanticMatcher] sentence-transformers is not installed, "
                  "using TF-IDF")
            self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            self._source_name_embs = self._vectorizer.fit_transform(names)
            self._use_bert = False

        return self

    def match(self, X_target: np.ndarray, target_col_names: list,
              verbose: bool = True, dataset_name=None) -> np.ndarray:

        assert self.source_stats is not None, "Error with fit"

        target_stats = compute_feature_stats(_normalize_for_stats(X_target))

        print("[match] target_stats (skew, kurt):")
        for i, name in enumerate(target_col_names[:5]):
            print(f"  {name}: skew={target_stats[i][2]:.3f}, kurt={target_stats[i][3]:.3f}")

        target_names = self.get_enriched_names(target_col_names, dataset_name)

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
        n_t_num = len(target_stats)

        for i in range(n_t):
            for j in range(n_s):
                if i < n_t_num and self.source_stats[j].any():
                    t = target_stats[i]
                    s = self.source_stats[j]
                    t_n = (t - t.mean()) / (t.std() + 1e-8)
                    s_n = (s - s.mean()) / (s.std() + 1e-8)
                    val = 1.0 / (1.0 + np.linalg.norm(t_n - s_n))
                    dist_sim[i, j] = 0.0 if np.isnan(val) else val

        similarity = self.alpha * sem_sim + self.beta * dist_sim
        for i in range(n_t_num, n_t):
            similarity[i] = sem_sim[i]

        age_idx = target_col_names.index('age') if 'age' in target_col_names else None
        if age_idx is not None:
            print(f"\n[debug] age similarities:")
            for j, src_name in enumerate(self.source_col_names):
                if sem_sim[age_idx, j] > 0.3:
                    print(
                        f"  {src_name}: sem={sem_sim[age_idx, j]:.3f}, dist={dist_sim[age_idx, j]:.3f}, total={similarity[age_idx, j]:.3f}")

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

    def _compute_similarity(self, X_target, target_col_names, dataset_name=None):
        from sklearn.metrics.pairwise import cosine_similarity

        target_names = self.get_enriched_names(target_col_names, dataset_name)

        if self._use_bert:
            target_embs = self._bert.encode(target_names, show_progress_bar=False)
        else:
            target_embs = self._vectorizer.transform(target_names)

        sem_sim = cosine_similarity(target_embs, self._source_name_embs)

        target_stats = compute_feature_stats(_normalize_for_stats(X_target))
        n_t = len(target_col_names)
        n_s = len(self.source_col_names)
        dist_sim = np.zeros((n_t, n_s), dtype=np.float32)

        for i in range(len(target_stats)):
            for j in range(n_s):
                if self.source_stats[j].any():
                    t_n = (target_stats[i] - target_stats[i].mean()) / (target_stats[i].std() + 1e-8)
                    s_n = (self.source_stats[j] - self.source_stats[j].mean()) / (self.source_stats[j].std() + 1e-8)
                    val = 1.0 / (1.0 + np.linalg.norm(t_n - s_n))
                    dist_sim[i, j] = 0.0 if np.isnan(val) else val

        similarity = self.alpha * sem_sim + self.beta * dist_sim
        return similarity.astype(np.float32)

    def match_hungarian(self, X_target, target_col_names, min_similarity=0.5, dataset_name=None):
        similarity = self._compute_similarity(X_target, target_col_names, dataset_name=dataset_name)

        # linear_sum_assignment minimize — invert
        cost_matrix = 1.0 - similarity  # [n_target, n_source]

        # Hungarian algorithn
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        result = {}
        for i, j in zip(row_ind, col_ind):
            sim = similarity[i, j]

            if sim < min_similarity:
                print(f"  ✗ {target_col_names[i]:25s} → not found (max={sim:.3f})")
                continue

            flag = "✓" if sim > 0.6 else "~"
            print(f"  {flag} {target_col_names[i]:25s} → {self.source_col_names[j]:25s} (total={sim:.3f})")
            result[target_col_names[i]] = self.source_col_names[j]

        return result  # {new_col: old_col} only correct features

    def get_match_indices_hungarian(self, X_target, num_col_names,
                                    cat_col_names, min_similarity=0.5, dataset_name=None):

        all_cols = num_col_names + cat_col_names

        matched = self.match_hungarian(X_target, all_cols, min_similarity, dataset_name=dataset_name)

        match_indices = {'num': {}, 'cat': {}}
        n_num = len(num_col_names)

        for i, col in enumerate(all_cols):
            if col not in matched:
                continue
            old_col = matched[col]
            j = self.source_col_names.index(old_col)

            if i < n_num:
                match_indices['num'][i] = j
            else:
                match_indices['cat'][i - n_num] = j

        print(f"\nMatching: {len(match_indices['num'])} num, "
              f"{len(match_indices['cat'])} cat "
              f"из {len(all_cols)} всего")

        return match_indices

    def get_match_indices(self, X_target, num_col_names, cat_col_names):
        """Return {'num': {i: j}, 'cat': {i: j}} — mapping target→source"""

        target_stats = compute_feature_stats(_normalize_for_stats(X_target))
        all_col_names = num_col_names + cat_col_names
        target_names = [n.lower().replace("_", " ") for n in all_col_names]

        if self._use_bert:
            target_embs = self._bert.encode(target_names, show_progress_bar=False)
        else:
            target_embs = self._vectorizer.transform(target_names)

        sem_sim = cosine_similarity(target_embs, self._source_name_embs)
        n_t_num = len(num_col_names)
        n_t = len(all_col_names)
        n_s = len(self.source_col_names)

        dist_sim = np.zeros((n_t, n_s), dtype=np.float32)
        for i in range(n_t_num):
            for j in range(n_s):
                if self.source_stats[j].any():
                    t_n = (target_stats[i] - target_stats[i].mean()) / (target_stats[i].std() + 1e-8)
                    s_n = (self.source_stats[j] - self.source_stats[j].mean()) / (self.source_stats[j].std() + 1e-8)
                    val = 1.0 / (1.0 + np.linalg.norm(t_n - s_n))
                    dist_sim[i, j] = 0.0 if np.isnan(val) else val

        similarity = self.alpha * sem_sim + self.beta * dist_sim
        for i in range(n_t_num, n_t):
            similarity[i] = sem_sim[i]

        match_indices = {'num': {}, 'cat': {}}
        for i in range(n_t):
            j = int(np.argmax(similarity[i]))
            if i < n_t_num:
                match_indices['num'][i] = j
            else:
                match_indices['cat'][i - n_t_num] = j

        return match_indices

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

    @classmethod
    def merge(cls, matchers: list, alpha: float = 0.8, beta: float = 0.2) -> "FeatureSemanticMatcher":
        """
        Example:
            matcher_hc  = FeatureSemanticMatcher.load("tabtransformer_best_matcher.pkl")
            matcher_gmc = FeatureSemanticMatcher.load("tabtransformer_finetune_none_matcher.pkl")
            meta = FeatureSemanticMatcher.merge([matcher_hc, matcher_gmc])
        """
        meta = cls(alpha=alpha, beta=beta)
        meta._use_bert = True

        seen_cols = set()
        all_col_names = []
        all_stats = []
        all_embs = []

        for m in matchers:
            for i, col in enumerate(m.source_col_names):
                if col in seen_cols:
                    continue
                seen_cols.add(col)
                all_col_names.append(col)
                all_stats.append(m.source_stats[i])
                all_embs.append(m._source_name_embs[i])

        meta.source_col_names = all_col_names
        meta.source_stats = np.vstack(all_stats)
        meta._source_name_embs = np.vstack(all_embs)
        meta._bert = matchers[0]._bert  # BERT from first matcher'а

        print(f"[MetaMatcher] merged {len(matchers)} matchers → "
              f"{len(all_col_names)} unique features "
              f"(from {[len(m.source_col_names) for m in matchers]})")

        return meta

    @staticmethod
    def get_enriched_names(col_names: list, dataset_name: str = None) -> list:
        try:
            from .feature_descriptions import get_enriched_names
            return get_enriched_names(col_names, dataset_name)
        except ImportError:
            return [n.lower().replace("_", " ") for n in col_names]

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
        self.adapt_epochs_ = params.get("adapt_epochs", 5)
        self.dataset_name_ = params.get("dataset_name", None)


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
            "lambda_align": self.params.get("lambda_align", 0.0),
            "freeze_mode": self.params.get("freeze_mode", "full")
        }

        y_train_arr = y_train.values if hasattr(y_train, "values") else y_train
        y_val_arr = y_val.values if hasattr(y_val, "values") else y_val

        print("cardinalities: ", self.cardinalities_)

        if self.transfer_mode_ in (None, "pretrain", "finetune"):
            # Fit matcher on current dataset
            # self.matcher_ = FeatureSemanticMatcher()
            self.matcher_ = FeatureSemanticMatcher(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"  # ← FinBERT от Prosus
            )
            num_for_match = num_train if num_train is not None else cat_train
            # col_names = (self.num_features_ or []) + (self.cat_features_ or [])
            # self.matcher_ = FeatureSemanticMatcher()
            num_col_names = [c for c in (self.num_features_ or []) if c != 'Unnamed: 0']
            cat_col_names = self.cat_features_ or []

            if num_for_match is not None and num_col_names:
                self.matcher_.fit(num_for_match, num_col_names, dataset_name=self.dataset_name_)

                # categorical features only  BERT, stats = zeros
                if cat_col_names and self.matcher_._use_bert:
                    cat_names = FeatureSemanticMatcher.get_enriched_names(cat_col_names, self.dataset_name_)
                    cat_embs = self.matcher_._bert.encode(
                        cat_names, show_progress_bar=False)

                    # add source: name + NaN statistic + BERT
                    self.matcher_.source_col_names += cat_col_names
                    self.matcher_.source_stats = np.vstack([
                        self.matcher_.source_stats,
                        np.zeros((len(cat_col_names), 8), dtype=np.float32)
                    ])
                    self.matcher_._source_name_embs = np.vstack([
                        self.matcher_._source_name_embs,
                        cat_embs
                    ])

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
                    self.matcher_.alpha = 0.8
                    self.matcher_.beta = 0.2
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
            # Stage finetune with transfer of backbone
            assert self.backbone_path_,  "transfer_mode='finetune' требует backbone_path в параметрах"
            matcher_path = self.backbone_path_.replace(".pth", "_matcher.pkl")
            matcher_a = None

            if os.path.exists(matcher_path):
                matcher_current = FeatureSemanticMatcher.load(matcher_path)
                matcher_current.beta = 0.2

                all_matchers = [matcher_current]
                best_matcher_path = os.path.join(
                    os.path.dirname(self.backbone_path_),
                    "tabtransformer_best_matcher.pkl"
                )
                if os.path.exists(best_matcher_path) and best_matcher_path != matcher_path:
                    matcher_best = FeatureSemanticMatcher.load(best_matcher_path)
                    matcher_best.alpha = 0.8
                    matcher_best.beta = 0.2
                    all_matchers.append(matcher_best)

                if len(all_matchers) > 1:
                    matcher_a = FeatureSemanticMatcher.merge(all_matchers)
                else:
                    matcher_a = matcher_current

                print(f"Matcher source cols ({len(matcher_a.source_col_names)}): {matcher_a.source_col_names[:5]}")

                print(f"[Matcher] source_cols ({len(matcher_a.source_col_names)}): {matcher_a.source_col_names}")
                print(f"[Matcher] source_stats shape: {matcher_a.source_stats.shape}")

                skip_cols = {'Unnamed: 0'}
                num_features_clean = [c for c in (self.num_features_ or []) if c not in skip_cols]
                col_names_b = num_features_clean + (self.cat_features_ or [])

                if num_train is not None:
                    keep_idx = [i for i, c in enumerate(self.num_features_ or []) if c not in skip_cols]
                    num_for_match = num_train[:, keep_idx]
                else:
                    num_for_match = cat_train

                print("\n=== Feature mapping: Stage A → Stage B ===")
                matcher_a.match(num_for_match, col_names_b, verbose=True, dataset_name=self.dataset_name_)

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
                matcher_a=matcher_a if os.path.exists(matcher_path) else self.matcher_,
                num_features=self.num_features_,
                cat_features=self.cat_features_,
            )

        elif self.transfer_mode_ in ("zero_shot", "proj_adapt"):

            assert self.backbone_path_, "transfer_mode='zero_shot'/'proj_adapt' requires backbone_path"

            matcher_for_adapt = self.matcher_  # fallback

            matcher_path = self.backbone_path_.replace(".pth", "_matcher.pkl")
            if os.path.exists(matcher_path):
                matcher_current = FeatureSemanticMatcher.load(matcher_path)
                matcher_current.alpha = 0.8
                matcher_current.beta = 0.2

                all_matchers = [matcher_current]

                # Stage A matcher (tabtransformer_best_matcher.pkl)
                best_matcher_path = os.path.join(
                    os.path.dirname(self.backbone_path_),
                    "tabtransformer_best_matcher.pkl"
                )
                if os.path.exists(best_matcher_path) and best_matcher_path != matcher_path:
                    matcher_best = FeatureSemanticMatcher.load(best_matcher_path)
                    matcher_best.alpha = 0.8
                    matcher_best.beta = 0.2
                    all_matchers.append(matcher_best)

                # concat if many
                if len(all_matchers) > 1:
                    matcher_for_adapt = FeatureSemanticMatcher.merge(all_matchers)
                else:
                    matcher_for_adapt = matcher_current

                print(f"[Matcher] loaded {len(all_matchers)} matchers, "
                      f"total source cols: {len(matcher_for_adapt.source_col_names)}")

            self.model_, _ = adapt_to_new_dataset(
                cat_data=cat_train, num_data=num_train,
                y_data=y_train_arr,
                cardinalities=self.cardinalities_ or [],
                TT=TT_config,
                backbone_checkpoint=self.backbone_path_,
                mode=self.transfer_mode_,
                adapt_epochs=self.adapt_epochs_,
                device=self.device_,
                num_features=self.num_features_,
                cat_features=self.cat_features_,
                matcher=matcher_for_adapt,
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
        start = time.time()

        with torch.no_grad():
            for i, batch in enumerate(loader):

                cat = batch["cat"].to(self.device_) if "cat" in batch else None
                num = batch["num"].to(self.device_) if "num" in batch else None
                preds = self.model_(cat, num)
                preds = torch.sigmoid(preds)
                all_preds.append(preds.cpu().numpy())

                if (i + 1) % max(1, n_batches // 5) == 0:
                    elapsed = time.time() - start
                    rows_done = (i + 1) * self.params.get("batch_size", 128)
                    speed = rows_done / elapsed if elapsed > 0 else 0
                    eta = (n_batches - i - 1) * elapsed / (i + 1)
                    print(f"[predict_proba] {i + 1}/{n_batches} batch | "
                          f"{speed:.0f} rows/sec | ETA: {eta:.1f}s")

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