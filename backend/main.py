import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

app = FastAPI()

@app.post("/run_learning/")
async def run_learn(file: UploadFile = File(...)):
    """
        post for load and parse YAML configuration from UploadFile.

        Args:
            file (UploadFile): YAML file with configuration parser

        Returns:
            dict: loaded configuration
        """
    try:
        contents = await file.read()

        content_str = contents.decode('utf-8')

        cfg = yaml.safe_load(content_str)

        print(f"YAML configuration succesfully loaded from: {file.filename}")

        return run_main(cfg)

    except UnicodeDecodeError as e:
        print(f"Error during decoding uploaded file : {e}")
        raise
    except yaml.YAMLError as e:
        print(f"Error during parsing YAML: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def run_main(run_cfg: dict):
    from backend.data_explorer.data_explorer import get_preprocessor, split_data, apply_data
    # from .transformers_ import fit_tabtransformer
    from backend.baselines.baselines import train_ensemble_model
    import os
    import mlflow
    import pandas as pd
    print("start calculation...")
    try:

        NAME = os.getenv("user_name")
        PASS = os.getenv("user_pass")
        HOST = os.getenv("host", "localhost")
        PORT = os.getenv("port", 5432)
        DB = os.getenv("data_base", "postgres")
        SCHEMA = os.getenv("schema", "public")

        print(NAME, PASS, HOST, PORT, DB, SCHEMA)
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

        mlflow.set_tracking_uri(
            f"postgresql+psycopg2://{NAME}:{PASS}@{HOST}:{PORT}/{DB}?options=-csearch_path={SCHEMA}")

        X, y, num_features, cat_features = apply_data(type_data=TYPE_DATA,data=full_data_cfg) if full_data_cfg is not None else apply_data(type_data=TYPE_DATA)
        # df = add_features(df)
        print("trouble before split")
        assert isinstance(X, pd.DataFrame), "X must be DataFrame"

        X_train, X_test, y_train, y_test = split_data(target_col=y, df=X, cfg_split=SPLIT_CONFIG, method=SPLIT_TYPE)
        # preprocessor = get_preprocessor(X_train)

        #todo asserts to check valid of dataframes
        # assert isinstance(X_train, pd.DataFrame), "X_train must be DataFrame"
        # assert isinstance(X_test, pd.DataFrame), "X_test must be DataFrame"

        data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "cat_features": cat_features, "num_features": num_features}

        #todo carefully check each dtype from each part of learnin model
        print("Prerun checking dtypes of columns from each part of learning...")

        if full_data_cfg is not None and model_cfg is not None:
            train_ensemble_model(type_class=TYPE_CLASS, data=data, model=model_cfg, metrics=metrics,
                                 type_class_model=type_class_model)
        else:
            train_ensemble_model(type_data=TYPE_DATA, type_class=TYPE_CLASS, split_type=SPLIT_TYPE)

        return {"status": "success"}

    except Exception as e:
        print(e)
        return {"status": "error"}

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", reload=True)

    # df = load_data("/Users/danilalipatov/Credit_Scoring_Model_w_transformers/backend/datasets/german_credit_risk/german_credit_risk.csv")

    # cfg_path = Path(
    #     r"C:\Users\Danch\PycharmProjects\Credit_Scoring_Model_w_transformers\data\test_yaml.yaml"
    # )
    #
    # with cfg_path.open("r", encoding="utf-8") as f:
    #     cfg = yaml.safe_load(f)
    #
    # run_cfg = cfg

    # models = get_baseline_models()
    # results = {}
    #
    # for name, model in models.items():
    #     results[name] = run_model(preprocessor, X_train, X_test, y_train, y_test, model, name)
