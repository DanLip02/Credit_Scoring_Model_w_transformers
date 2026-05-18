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
from sklearn.feature_extraction.text import TfidfVectorizer
from .tab_transformers import compute_feature_stats, _normalize_for_stats
import time
import glob


class FeatureSemanticMatcher:
    """
    fit(X_source, col_names)  —  remember the statistics and feature names of the A dataset
    match(X_target, col_names) — return the similarity matrix [n_target, n_source]
    """

    def __init__(self, alpha: float = 0.8, beta: float = 0.2, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        # beta > alpha — distribution is more important than name
        self.alpha = alpha
        self.beta = beta
        self.source_stats = None
        self.source_col_names = None
        self._vectorizer = None
        self._source_name_embs = None
        self._bert = SentenceTransformer(model_name, model_kwargs={"use_safetensors": True})

    #todo add BERT for semantic.
    def fit(self, X_source: np.ndarray, col_names: list, dataset_name=None, backbone_path=None) -> "FeatureSemanticMatcher":

        self.source_col_names = list(col_names)
        self.source_stats = compute_feature_stats(_normalize_for_stats(X_source))
        self.backbone_path = backbone_path
        self.source_dataset_name = dataset_name
        self.source_col_indices = {col: i for i, col in enumerate(col_names)}

        names = self.get_enriched_names(col_names, dataset_name)
        print(f"[fit] example: '{col_names[0]}' → '{names[0]}'")

        try:
            print("[FeatureSemanticMatcher] BERT (multilingual)...")
            self._bert = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2", model_kwargs={"use_safetensors": True}
            )
            self._source_name_embs = self._bert.encode(names, show_progress_bar=False)  # [n_features, 384]

            norms = np.linalg.norm(self._source_name_embs, axis=1, keepdims=True)
            self._source_name_embs = self._source_name_embs / (norms + 1e-8)
            print(f"[fit] emb norm after normalize: {np.linalg.norm(self._source_name_embs[0]):.3f}")

            self._use_bert = True
        except ImportError:
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

                    if target_col_names[i] == 'Score_bki' and self.source_col_names[j] == 'bureau_bb_months_min':
                        print(f"[debug] Score_bki vs bureau_bb_months_min:")
                        print(f"  t_stats: {target_stats[i]}")
                        print(f"  s_stats: {self.source_stats[j]}")
                        print(f"  t_n: {t_n}")
                        print(f"  s_n: {s_n}")
                        print(f"  dist: {np.linalg.norm(t_n - s_n):.4f}")
                        print(f"  val: {val:.4f}")

        similarity = self.alpha * sem_sim + self.beta * dist_sim

        self._last_sem_sim = sem_sim
        self._last_dist_sim = dist_sim

        return similarity.astype(np.float32)

    def match_hungarian(self, X_target, target_col_names, min_similarity=0.5, dataset_name=None):
        similarity = self._compute_similarity(X_target, target_col_names, dataset_name=dataset_name)

        # linear_sum_assignment minimize — invert
        cost_matrix = 1.0 - similarity  # [n_target, n_source]

        # Hungarian algorithn
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        if 'age' in target_col_names:
            i = target_col_names.index('age')
            top5 = np.argsort(similarity[i])[::-1][:5]
            print(f"[debug] age top-5 before Hungarian:")
            for j in top5:
                print(f"  {self.source_col_names[j]}: {similarity[i, j]:.3f}")

        result = {}
        for i, j in zip(row_ind, col_ind):
            sim = similarity[i, j]
            sem = self._last_sem_sim[i, j]
            dist = self._last_dist_sim[i, j]

            if sim < min_similarity:
                best_j = np.argmax(self._last_sem_sim[i])
                print(f"  ✗ {target_col_names[i]:25s} → not found "
                      f"(max_total={sim:.3f}, max_sem={self._last_sem_sim[i, j]:.3f}, "
                      f"max_stat={self._last_dist_sim[i, j]:.3f})")
                continue

            min_sem = 0.5
            if sem < min_sem:
                print(f"  ✗ {target_col_names[i]:25s} → sem too low "
                      f"(sem={sem:.3f} < {min_sem}, total={sim:.3f})")
                continue

            flag = "✓" if sim > 0.65 else "~"
            print(f"  {flag} {target_col_names[i]:25s} → {self.source_col_names[j]:25s} "
                  f"(total={sim:.3f}, sem={self._last_sem_sim[i, j]:.3f}, stat={self._last_dist_sim[i, j]:.3f})")
            result[target_col_names[i]] = self.source_col_names[j]

        return result  # {new_col: old_col} only correct features

    def get_match_indices_hungarian(self, X_target, num_col_names,
                                    cat_col_names, min_similarity=0.72, dataset_name=None):

        print(f"[get_match_indices_hungarian] dataset_name={dataset_name}")

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
        with open(path, "rb") as f:
            obj = pickle.load(f)

        print(f"[Matcher.load] source_col_names[:3]: {obj.source_col_names[:3]}")
        if hasattr(obj, 'source_dataset_name'):
            print(f"[Matcher.load] source_dataset_name: {obj.source_dataset_name}")
        else:
            print(f"[Matcher.load] source_dataset_name: NOT SET (old pkl without descriptions)")

        if 'DAYS_BIRTH' in obj.source_col_names:
            idx = obj.source_col_names.index('DAYS_BIRTH')
            emb = obj._source_name_embs[idx]
            print(f"[Matcher.load] DAYS_BIRTH emb[:5]: {emb[:5]}")
            print(f"[Matcher.load] DAYS_BIRTH emb norm: {np.linalg.norm(emb):.3f}")

        return obj

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

        meta.source_col_to_backbone = {}

        for m in matchers:
            for i, col in enumerate(m.source_col_names):
                if col in seen_cols:
                    continue
                seen_cols.add(col)
                all_col_names.append(col)
                all_stats.append(m.source_stats[i])
                all_embs.append(m._source_name_embs[i])

                meta.source_col_to_backbone[col] = {
                    "backbone_path": getattr(m, "backbone_path", None),
                    "local_idx": getattr(m, "source_col_indices", {}).get(col, i)
                }

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
            from .feature_descriptions import get_enriched_names as _get_enriched
            result = _get_enriched(col_names, dataset_name)
            if dataset_name:
                print(f"[enriched_names] dataset={dataset_name}, example:")
                print(f"  '{col_names[0]}' → '{result[0]}'")
            else:
                print(f"[enriched_names] no dataset_name → plain names")
                print(f"  '{col_names[0]}' → '{result[0]}'")
            return result
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
        self.matcher_ = None


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
            "freeze_mode": self.params.get("freeze_mode", "full"),
            "dataset_name": self.params.get("dataset_name", None),
            "model_name": self.params.get("model_name", "tabtransformer_finetune_none")
        }

        y_train_arr = y_train.values if hasattr(y_train, "values") else y_train
        y_val_arr = y_val.values if hasattr(y_val, "values") else y_val

        print("cardinalities: ", self.cardinalities_)

        if self.transfer_mode_ in (None, "pretrain", "finetune"):
            # Fit matcher on current dataset
            # self.matcher_ = FeatureSemanticMatcher()
            self.matcher_ = FeatureSemanticMatcher(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
            num_for_match = num_train if num_train is not None else cat_train
            # col_names = (self.num_features_ or []) + (self.cat_features_ or [])
            # self.matcher_ = FeatureSemanticMatcher()
            num_col_names = [c for c in (self.num_features_ or []) if c != 'Unnamed: 0']
            cat_col_names = self.cat_features_ or []

            if num_for_match is not None and num_col_names:

                model_name = self.params.get("model_name", "tabtransformer_finetune_none")
                models_dir = os.path.dirname(self.backbone_path_) if self.backbone_path_ else os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "transformers_"
                )
                current_model_path = os.path.join(models_dir, f"{model_name}.pth")

                self.matcher_.fit(num_for_match, num_col_names, dataset_name=self.dataset_name_, backbone_path=current_model_path)

                # categorical features only  BERT, stats = zeros
                if cat_col_names and self.matcher_._use_bert:
                    cat_names = FeatureSemanticMatcher.get_enriched_names(cat_col_names, self.dataset_name_)
                    cat_embs = self.matcher_._bert.encode(
                        cat_names, show_progress_bar=False)

                    # add source: name + NaN statistic + BERT
                    self.matcher_.source_col_names += cat_col_names
                    self.matcher_.source_stats = np.vstack([
                        self.matcher_.source_stats,
                        np.zeros((len(cat_col_names), 9), dtype=np.float32)
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

                model_name = self.params.get("model_name", "tabtransformer_stageA_hc")
                BASE_DIR = os.path.dirname(os.path.abspath(__file__))

                if self.backbone_path_:
                    models_dir = os.path.dirname(self.backbone_path_)
                else:
                    models_dir = os.path.normpath(
                        os.path.join(BASE_DIR, "..", "..", "models", "transformers_")
                    )

                matcher_path = os.path.join(models_dir, f"{model_name}_matcher.pkl")

                self.matcher_.save(matcher_path)
                print(f"[Matcher] saved to: {matcher_path}")

        if self.transfer_mode_ in (None, "pretrain"):
            self.model_ = fit_tabtransformer(
                cat_train=cat_train, num_train=num_train,
                y_train=y_train_arr,
                cat_val=cat_val, num_val=num_val,
                y_val=y_val_arr,
                cardinalities=self.cardinalities_ or [],
                TT=TT_config,
                device=self.device_,
                num_features=self.num_features_,
                cat_features=self.cat_features_
            )

        elif self.transfer_mode_ == "finetune":
            # Stage finetune with transfer of backbone
            assert self.backbone_path_,  "transfer_mode='finetune' requires backbone_path in params"
            matcher_a = None
            models_dir = os.path.dirname(self.backbone_path_)
            all_matcher_files = sorted(glob.glob(
                os.path.join(models_dir, "tabtransformer_stage*_matcher.pkl")
            ))
            print(f"[glob] found: {all_matcher_files}")

            all_matcher_files = [p for p in all_matcher_files
                                 if model_name not in os.path.basename(p)]

            if all_matcher_files:
                all_matchers = []
                for p in all_matcher_files:
                    m = FeatureSemanticMatcher.load(p)
                    m.alpha = 0.8
                    m.beta = 0.2
                    all_matchers.append(m)
                matcher_a = FeatureSemanticMatcher.merge(all_matchers)
                print(f"[Matcher] loaded {len(all_matchers)} matchers → "
                      f"{len(matcher_a.source_col_names)} cols")
            else:
                matcher_a = self.matcher_

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
                matcher_a=matcher_a if matcher_a is not None else self.matcher_,
                num_features=self.num_features_,
                cat_features=self.cat_features_,
            )

        elif self.transfer_mode_ in ("zero_shot", "proj_adapt"):

            assert self.backbone_path_, "transfer_mode='zero_shot'/'proj_adapt' requires backbone_path"

            matcher_for_adapt = self.matcher_  # fallback
            models_dir = os.path.dirname(self.backbone_path_)
            all_matcher_files = sorted(glob.glob(
                os.path.join(models_dir, "tabtransformer_stage*_matcher.pkl")
            ))

            if all_matcher_files:
                all_matchers = []
                for p in all_matcher_files:
                    m = FeatureSemanticMatcher.load(p)
                    m.alpha = 0.8
                    m.beta = 0.2
                    all_matchers.append(m)
                matcher_for_adapt = FeatureSemanticMatcher.merge(all_matchers)
                print(f"[Matcher] loaded {len(all_matchers)} matchers → "
                      f"{len(matcher_for_adapt.source_col_names)} cols")
            else:
                print(f"[Matcher] no stage matchers found in {models_dir}")

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
                f"Not correct transfer_mode: {self.transfer_mode_!r}. "
                f"Available: None, 'pretrain', 'finetune', 'zero_shot', 'proj_adapt'"
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