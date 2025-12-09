import numpy as np
import requests
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from dotenv import load_dotenv
import json
import os


load_dotenv()

def compute_ar_metric(y_true, y_pred, skip=None,
                      use_transform="direct",
                      min_benchmark=1, max_benchmark=7,
                      min_score=1, max_score=7):

    URL_ar = os.getenv("URL_ar")
    payload = {
        "score": list(map(int, y_pred)),
        "ground_truth": list(map(int, y_true)),
        "skip": skip or [0] * len(y_true),
        "use_transform": use_transform,
        "min_benchmark": min_benchmark,
        "max_benchmark": max_benchmark,
        "min_score": min_score,
        "max_score": max_score
    }

    response = requests.post(
        URL_ar,
        headers={"accept": "application/json", "Content-Type": "application/json"},
        data=json.dumps(payload),
        verify=False #todo add cert for this https
    )

    if response.status_code == 200:
        return response.json()["result"]
    else:
        raise ValueError(f"Error while init AR: {response.status_code}, {response.text}. Check data or VPN(access)")

def calc_base_metrics(metrics_list, y_test, y_pred, y_prob=None):
    results = {}
    for metric in metrics_list:
        try:
            if metric == "accuracy":
                results["accuracy"] = accuracy_score(y_test, y_pred)
            elif metric == "f1":
                results["f1"] = f1_score(y_test, y_pred, average="macro")
            elif metric == "auc":
                results["auc"] = roc_auc_score(y_test, y_prob if y_prob is not None else y_pred, multi_class='ovo')
            elif metric == "mse":
                results["mse"] = mean_squared_error(y_test, y_pred)
            elif metric == "mae":
                results["mae"] = mean_absolute_error(y_test, y_pred)
            elif metric == "r2":
                results["r2"] = r2_score(y_test, y_pred)
            else:
                print(f"Not correct: {metric}")
                results[metric] = None
        except Exception:
            results[metric] = None
    return results


def calc_custom_metrics(metrics_list, y_test, y_pred, y_prob=None):
    """
    Calculating cusom metric by name

    metrics_list: dict, keys - name of metrics, value — dict of params for each custom metric
    y_test: true values
    y_pred: forecasted values
    y_prob: prob of values (optional)
    """
    results = {}

    for metric_name, params in metrics_list.items():
        try:
            if metric_name == "ar":
                results[metric_name] = compute_ar_metric(
                    y_true=y_test,
                    y_pred=y_pred,
                    **params
                )
            else:
                print(f"[WARN] Metric '{metric_name}' not implemented.")
                results[metric_name] = None

        except Exception as e:
            print(f"[ERROR] Error computing metric '{metric_name}': {e}")
            results[metric_name] = None

    return results


def load_metric(metrics: dict, y_test, y_pred, y_prob=None, api_url=None, api_token=None):
    """
    Combine base and custom metrics
    """
    results = {}

    # base
    if "base" in metrics:
        results.update(calc_base_metrics(metrics["base"], y_test, y_pred, y_prob))

    # custom
    if "custom" in metrics:
        results.update(calc_custom_metrics(metrics["custom"], y_test, y_pred, y_prob))

    return results