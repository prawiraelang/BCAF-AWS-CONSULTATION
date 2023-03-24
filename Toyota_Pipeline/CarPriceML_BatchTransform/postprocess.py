# Postprocess for Batch Transform With Constant
import argparse
import logging
import os
import pathlib
import boto3
import pandas as pd

from time import gmtime, strftime

'''
Add your required additional dependencies here!
'''

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.debug("Starting postprocessing...")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-lelang", type=str, required=True)
    parser.add_argument("--input-data-crawling", type=str, required=True)
    parser.add_argument("--input-batch-transform", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--prefix-batch-transform", type=str, required=True)
    
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/raw").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{base_dir}/batch_transform").mkdir(parents=True, exist_ok=True)
    input_data_lelang = args.input_data_lelang
    input_data_crawling = args.input_data_crawling
    input_batch_transform = args.input_batch_transform
    model_type = args.model_type
    prefix_batch_transform = args.prefix_batch_transform
    
    bucket_lelang = input_data_lelang.split("/")[2]
    key_lelang = "/".join(input_data_lelang.split("/")[3:])
    
    bucket_crawling = input_data_crawling.split("/")[2]
    key_crawling = "/".join(input_data_crawling.split("/")[3:])
    
    name_batch_out = "predict" # This variable MUST be the same as in preprocess.py
    name_file_send = "prediction_results"
    
    bucket_batch = input_batch_transform.split("/")[2]
    key_batch = "/".join(input_batch_transform.split("/")[3:]) + "/" + name_batch_out + ".csv.out"
    
    s3_singapore = boto3.resource("s3", region_name="ap-southeast-1")
    s3_virginia = boto3.resource("s3", region_name="us-east-1")
    
    logger.info("Downloading lelang data from <%s/%s>...", bucket_lelang, key_lelang)
    lelang_path = f"{base_dir}/raw/lelang.csv"
    s3_singapore.Bucket(bucket_lelang).download_file(key_lelang, lelang_path)
    
    logger.info("Downloading crawling data from <%s/%s>...", bucket_crawling, key_crawling)
    crawling_path = f"{base_dir}/raw/crawling.csv"
    s3_singapore.Bucket(bucket_crawling).download_file(key_crawling, crawling_path)
    
    logger.info("Downloading batch transform data from <%s/%s>...", bucket_batch, key_batch)
    batch_path = f"{base_dir}/raw/batch_transform.csv"
    s3_virginia.Bucket(bucket_batch).download_file(key_batch, batch_path)

    logger.info("Reading lelang data...")
    df_lelang = pd.read_csv(lelang_path)
    os.unlink(lelang_path)
    
    logger.info("Reading crawling data...")
    df_crawling = pd.read_csv(crawling_path)
    os.unlink(crawling_path)
    
    logger.info("Reading batch transform data...")
    df_batch = pd.read_csv(batch_path, header=None, names=['prediksi'])
    os.unlink(batch_path)
    
    '''
    Add your own postprocessing step here!
    '''
    
    df = df_lelang # You need to join df_lelang and df_crawling after/before you postprocess it
    
    df.reset_index(drop=True, inplace=True)
    df_batch.reset_index(drop=True, inplace=True)

    df_result = pd.concat([df, df_batch], axis=1) 
    
    # unique_key = strftime("%Y%m%d-%H:%M:%S", gmtime())
    unique_key = strftime("%Y%m%d", gmtime())
    
    # Save the data to base directory
    logger.info("Writing out dataset to base directory...")
    df_result.to_csv(f"{base_dir}/batch_transform/batch_transform.csv", index=False)
    
    # Upload the data to S3
    logger.info("Writing out datasets to <%s>...", bucket_batch)
    s3_virginia.meta.client.upload_file(f"{base_dir}/batch_transform/batch_transform.csv", Bucket=bucket_batch, Key=f"{prefix_batch_transform}/{model_type}/with_header/{unique_key}/{name_file_send}.csv")