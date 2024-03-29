# Evaluate for Training
import json
import logging
import pathlib
import tarfile
import numpy as np
import pandas as pd
import xgboost
import boto3

from time import gmtime, strftime
from sklearn.metrics import mean_squared_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Starting evaluation...")
    s3_virginia = boto3.resource("s3", region_name="us-east-1")
    
    model_path = "/opt/ml/processing/model/model.tar.gz"
    unique_key = strftime("%Y%m%d", gmtime())
    default_bucket = "glair-exploration-sagemaker-bucket" # To save evaluation report
    
    with tarfile.open(model_path) as tar:
        tar.extractall(path="..")

    logger.info("Loading model...")
    model = xgboost.Booster()
    model.load_model("xgboost-model")

    logger.info("Reading test data...")
    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)

    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.values)

    logger.info("Performing predictions against test data...")
    predictions = model.predict(X_test)

    logger.info("Calculating evaluation metrics...")
    rmse = mean_squared_error(y_test, predictions, squared=False)
    std = np.std(y_test - predictions)
    report_dict = {
        "regression_metrics": {
            "rmse": {
                "value": rmse,
                "standard_deviation": std
            },
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Writing out evaluation report...")
    logger.info("RMSE is %f", rmse)
    logger.info("Standard deviaton is %f", std)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

    s3_virginia.meta.client.upload_file(f"{output_dir}/evaluation.json", Bucket=default_bucket, Key=f"glair-bcaf-consultation-output/evaluation/{unique_key}/evaluation.json")