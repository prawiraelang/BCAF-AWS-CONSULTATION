import argparse
import logging
import os
import pathlib
import boto3
import pandas as pd
from time import gmtime, strftime

from sklearn.model_selection import train_test_split
'''
Add your required additional dependencies here!
'''

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting preprocessing...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-lelang", type=str, required=True)
    parser.add_argument("--input-data-crawling", type=str, required=True)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/raw").mkdir(parents=True, exist_ok=True)
    input_data_lelang = args.input_data_lelang
    input_data_crawling = args.input_data_crawling
    bucket_lelang = input_data_lelang.split("/")[2]
    key_lelang = "/".join(input_data_lelang.split("/")[3:])
    bucket_crawling = input_data_crawling.split("/")[2]
    key_crawling = "/".join(input_data_crawling.split("/")[3:])

    logger.info("Downloading lelang data from <%s/%s>...", bucket_lelang, key_lelang)
    s3 = boto3.resource("s3")
    
    lelang_path = f"{base_dir}/raw/lelang.csv"
    s3.Bucket(bucket_lelang).download_file(key_lelang, lelang_path)
    
    logger.info("Downloading lelang data from <%s/%s>...", bucket_crawling, key_crawling)
    crawling_path = f"{base_dir}/raw/crawling.csv"
    s3.Bucket(bucket_crawling).download_file(key_crawling, crawling_path)

    logger.info("Reading lelang data...")
    df_lelang = pd.read_csv(lelang_path)
    os.unlink(lelang_path)
    
    logger.info("Reading crawling data...")
    df_crawling = pd.read_csv(crawling_path)
    os.unlink(crawling_path)
    
    '''
    Add your own preprocessing step here!
    '''
    
    df = df_lelang # You need to join df_lelang and df_crawling after/before you preprocess it

    logger.info("Splitting rows of joined data into train, validation, test sets...")
    # Separate the features and the target columns
    X = df.drop(df.columns[0], axis=1)
    y = df[df.columns[0]]
    
    # Splitting data into train, validation, and test sets
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, random_state=293)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=342)

    # Concatenate data
    # The target column must in the first column
    df_train = pd.concat([y_train, pd.DataFrame(X_train, index=X_train.index, columns=X_train.columns)], axis=1)
    df_val = pd.concat([y_val, pd.DataFrame(X_val, index=X_val.index, columns=X_val.columns)], axis=1)
    df_test = pd.concat([y_test, pd.DataFrame(X_test, index=X_test.index, columns=X_test.columns)], axis=1)

    unique_key = strftime("%Y%m%d-%H:%M:%S", gmtime())
    logger.info("Writing out datasets to <%s>...", bucket_crawling)
    df_train.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    df_val.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    df_test.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    
    # Upload the data to S3
    s3.meta.client.upload_file(f"{base_dir}/train/train.csv", Bucket=bucket_lelang, Key=f"glair-bcaf-consultation/training/train/{unique_key}/train.csv")
    s3.meta.client.upload_file(f"{base_dir}/validation/validation.csv", Bucket=bucket_lelang, Key=f"glair-bcaf-consultation/training/validation/{unique_key}/validation.csv")
    s3.meta.client.upload_file(f"{base_dir}/test/test.csv", Bucket=bucket_lelang, Key=f"glair-bcaf-consultation/training/test/{unique_key}/test.csv")
    s3.meta.client.upload_file(f"{base_dir}/test/test.csv", Bucket=bucket_lelang, Key=f"glair-bcaf-consultation/batch_transform/input/{unique_key}/predict.csv")
