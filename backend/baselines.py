#### Module with basic models for credit risk modeling
from sklearn.ensemble import (
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, cross_val_score
"""
There will be some models that are really popular in credit risk modeling

For ex.

Gradient boosting: Catboost, XGBoost, LGBMBoost and others

Bagging: RandomForest, BalancedRandomForest

Stacking: Boosting + Bagging + Linear Model

Voting Classifier: Boosting + Bagging + Stacking + others

Logisticregression / LinearRegression / RidgeRegression / LasssoRegression and others

"""


def run_model(preprocessor, X_train, X_test, y_train, y_test, model, name="model", use_smote=False):
    if use_smote:
        pipe = ImbPipeline([
            ("preproc", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("clf", model)
        ])
    else:
        pipe = Pipeline([
            ("preproc", preprocessor),
            ("clf", model)
        ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print(f"\n=== {name} ===")
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    # --- независимая метрика: KFold AUC ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=kf,
                             scoring="roc_auc", n_jobs=-1)
    print(f"AUC (5-fold CV): mean={cv_auc.mean():.4f} | std={cv_auc.std():.4f}")
    print(classification_report(y_test, y_pred))
    return {"AUC": auc, "Report": classification_report(y_test, y_pred, output_dict=True)}

def get_baseline_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=500, learning_rate=0.03, class_weight="balanced", random_state=42)
    }
    return models

