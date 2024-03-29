# Preprocess for Batch Transform With Constant
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
    logger.info("Starting preprocessing...")
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
    
    name_batch_out = "predict" # This variable MUST be the same as in postprocess.py
    
    s3_singapore = boto3.resource("s3", region_name="ap-southeast-1")
    
    logger.info("Downloading lelang data from <%s/%s>...", bucket_lelang, key_lelang)
    lelang_path = f"{base_dir}/raw/lelang.csv"
    s3_singapore.Bucket(bucket_lelang).download_file(key_lelang, lelang_path)
    
    logger.info("Downloading crawling data from <%s/%s>...", bucket_crawling, key_crawling)
    crawling_path = f"{base_dir}/raw/crawling.csv"
    s3_singapore.Bucket(bucket_crawling).download_file(key_crawling, crawling_path)

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

    # unique_key = strftime("%Y%m%d-%H:%M:%S", gmtime())
    unique_key = strftime("%Y%m%d", gmtime())
    
    # Save the data to base directory
    logger.info("Writing out dataset to base directory...")
    df.to_csv(f"{base_dir}/predict/{name_batch_out}.csv", header=False, index=False)