from data_explorer import *
from baselines import run_model, get_baseline_models





if __name__ == '__main__':
    df = load_data("/Users/danilalipatov/Credit_Scoring_Model_w_transformers/backend/datasets/german_credit_risk/german_credit_risk.csv")
    df = add_features(df)
    print(df.columns)
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = get_preprocessor(X_train)

    models = get_baseline_models()
    results = {}

    for name, model in models.items():
        results[name] = run_model(preprocessor, X_train, X_test, y_train, y_test, model, name)
