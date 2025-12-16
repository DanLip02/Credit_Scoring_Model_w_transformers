from .data_explorer.data_explorer import *
from .transformers import fit_tabtransformer
from .baselines import train_ensemble_model
import torch
import os
import yaml

if __name__ == '__main__':
    # df = load_data("/Users/danilalipatov/Credit_Scoring_Model_w_transformers/backend/datasets/german_credit_risk/german_credit_risk.csv")

    cfg_path = Path(
        r"C:\Users\Danch\PycharmProjects\Credit_Scoring_Model_w_transformers\data\test_yaml.yaml"
    )

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_cfg = cfg

    TYPE_DATA = run_cfg["run"]["type_data"]
    SPLIT_TYPE = run_cfg["run"]["split_type"]
    SPLIT_CONFIG = run_cfg["run"]["split_config"]
    TYPE_CLASS = run_cfg["run"]["type_class"]

    full_data_cfg = run_cfg.get("full_data", None)
    model_cfg = run_cfg.get("model", None)
    metrics = run_cfg.get("metrics", None)

    # todo check name_experiment work in process
    name_experiment = model_cfg.get("model_name", None) if model_cfg is not None else None
    type_class_model = run_cfg.get("type_class_model", None)


    X, y, num_features, cat_features = apply_data(type_data=TYPE_DATA, data=full_data_cfg) if full_data_cfg is not None else apply_data(type_data=TYPE_DATA)
    # df = add_features(df)

    X_train, X_test, y_train, y_test = split_data(target_col=y, df=X, cfg_split=SPLIT_CONFIG, method=SPLIT_TYPE)
    # preprocessor = get_preprocessor(X_train)

    data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    if full_data_cfg is not None and model_cfg is not None:
        train_ensemble_model(type_class=TYPE_CLASS, data=data, model=model_cfg, metrics=metrics, type_class_model=type_class_model)
    else:
        train_ensemble_model(type_data=TYPE_DATA, type_class=TYPE_CLASS, split_type=SPLIT_TYPE)

    stop = 0

    # models = get_baseline_models()
    # results = {}
    #
    # for name, model in models.items():
    #     results[name] = run_model(preprocessor, X_train, X_test, y_train, y_test, model, name)
