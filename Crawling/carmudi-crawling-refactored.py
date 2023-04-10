# Import libraries
import pandas as pd
import boto3
import json
import logging
import time

from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from time import gmtime, strftime
from io import StringIO

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

DEFAULT = 'default'
BRAND = 'toyota'
VARIANT = 'avanza'
COL_PRICE = 'price'

TRANSMISSION_MANUAL = 'manual'
TRANSMISSION_AUTOMATIC = 'automatic'

CAR_TYPE_G = 'G'
CAR_TYPE_VELOZ = 'VELOZ'
CAR_TYPE_E = 'E'
CAR_TYPE_S = 'S'

CC_1_3 = '1.3'
CC_1_5 = '1.5'

COL_TYPE = 'Type'
COL_TRANSMISI = 'transmisi'
COL_TYPE_DETAIL = 'type_detail'
COL_CC = 'cc'
COL_TAHUN = 'tahun'

map_transmission = {
    TRANSMISSION_MANUAL: ['M/T', 'MT'],
    TRANSMISSION_AUTOMATIC:['A/T', 'AT'],
    DEFAULT: TRANSMISSION_MANUAL
}

map_car_type = {
    CAR_TYPE_G: [' G'],
    CAR_TYPE_VELOZ: [' VELOZ'],
    CAR_TYPE_E: [' E'],
    CAR_TYPE_S: [' S'],
    DEFAULT: CAR_TYPE_E
}

map_cc = {
    CC_1_3: ['1.3'],
    CC_1_5: ['1.5'],
    DEFAULT: CC_1_3
}

def get_latest_file(s3_client, bucket_name, prefix_name):
    s3_uri_response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix_name)
    try:
        csv_keys = [obj for obj in s3_uri_response.get("Contents", []) if obj["Key"].endswith(".csv")]
        latest_file_key = sorted(csv_keys, key=lambda x: x["LastModified"], reverse=True)[0]["Key"]
    except:
        excel_keys = [obj for obj in s3_uri_response.get("Contents", []) if obj["Key"].endswith(".xlsx")]
        latest_file_key = sorted(excel_keys, key=lambda x: x["LastModified"], reverse=True)[0]["Key"]
    return f"s3://{bucket_name}/{latest_file_key}"

def get_lelang_dataframe(s3_client):
    s3_uri_lelang = get_latest_file(
        s3_client,
        "glair-exploration-sagemaker-s3-bucket-singapore",
        "glair-bcaf-consultation-input/training/toyota/lelang/"
    )

    if s3_uri_lelang.endswith(".csv"):
        bucket_lelang = s3_uri_lelang.split("/")[2]
        key_lelang = "/".join(s3_uri_lelang.split("/")[3:])
        file_object = s3_client.get_object(Bucket=bucket_lelang, Key=key_lelang)

        return pd.read_csv(file_object['Body'], sep=";")
    elif s3_uri_lelang.endswith(".xlsx"):
        return pd.read_excel(s3_uri_lelang)
    else:
        return "File extension is not supported! File extension must be .csv or .xlsx."

def get_lelang_simple():
    return pd.read_excel(f"bcaf-lelang.xlsx")

def mapping_label(value, dict_label):
    for key, labels in dict_label.items():
        if any(label in value for label in labels):
            return key
    return dict_label['default']

def get_column_mapping(df, origin_column, dict_label):
    return df[origin_column].apply(lambda value: mapping_label(value, dict_label))

def save_to_s3(s3_resource, df):
    unique_key = strftime("%Y%m%d", gmtime())
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    bucket = s3_resource.Bucket("glair-exploration-sagemaker-s3-bucket-singapore")
    return bucket.put_object(Key=f"training/toyota/crawling/{unique_key}/data-crawling-{unique_key}.csv", Body=csv_buffer.getvalue())

def crawling():
    # Get S3 client to get dataframe
    s3_client = boto3.client("s3", region_name="us-east-1")

    # logger.info("Read dataframe...")
    print("Reading dataframe...")

    # Get dataframe
    # df_lelang = get_lelang_dataframe(s3_client)
    df_lelang = get_lelang_simple()

    # Extract 'transmisi' from 'Type' column
    df_lelang[COL_TRANSMISI] = get_column_mapping(df_lelang, COL_TYPE, map_transmission)

    # Extract 'type_detail' from 'Type' column
    df_lelang[COL_TYPE_DETAIL] = get_column_mapping(df_lelang, COL_TYPE, map_car_type)

    # Extract 'cc' from 'Type' column
    df_lelang[COL_CC] = get_column_mapping(df_lelang, COL_TYPE, map_cc)

    df_lelang = df_lelang[['type_detail', 'cc', 'tahun', 'transmisi']]

    # Start crawling
    # logger.info("Starting the crawling process. The time it takes to complete will depend on the internet speed.")
    print("Starting the crawling process. The time it takes to complete will depend on the internet speed.")
    
    start_time = time.time() # Record the start time
    results = []
    for index, row in df_lelang.iterrows():
        alamat = f"https://www.carmudi.co.id/en/cars-for-sale/{BRAND}/{VARIANT}/year-{row[COL_TAHUN]}/indonesia?transmission={row[COL_TRANSMISI]}"
        req = Request(alamat, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req).read()
        data = BeautifulSoup(html, 'html.parser')
        script = data.find_all('script', {'type': 'application/ld+json'})[0]

        # print(json.loads(script.text))
        script_ = json.loads(script.text)
        # print(script_)
        
        # Check if crawling result exist
        if isinstance(script_, list) and len(script_) > 0 and 'itemListElement' in script_[0]:
            # logger.info(f"[FOUND] There is a crawling result found at index number {index} with the address at {alamat}")
            print(f"[FOUND] There is a crawling result found at index number {index} with the address at {alamat}")

            rows = script_[0]['itemListElement']

            for a in range(len(rows)):
                harga   = rows[a]['item']['offers']['price']
                row_hasil = {}
                row_hasil = {
                    COL_PRICE: harga,
                    COL_TAHUN: row[COL_TAHUN],
                    COL_CC: row[COL_CC],
                    COL_TYPE: row[COL_TYPE_DETAIL]
                }

                # Save results
                results.append(row_hasil)
        else:
            # logger.info(f"[NOT FOUND] No crawling result found at index number {index} with the address at {alamat}.")
            print(f"[NOT FOUND] No crawling result found at index number {index} with the address at {alamat}.")

        # For debug
        if index == 10:
            break
  
        # To mimic human behavior
        if index % 7 == 0:
            time.sleep(5)

    end_time = time.time() # Record the end time
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    # logger.info(f"Crawling process finished after: {elapsed_time:.2f} seconds")
    print(f"Crawling process finished after: {elapsed_time:.2f} seconds")

    df_crawling = pd.DataFrame(results)
    
    # Get S3 resource to save dataframe
    # s3_resource = boto3.resource("s3", region_name="us-east-1")

    # Save dataframe to S3
    # save_to_s3(s3_resource, df_crawling)

    # logger.info("Dataframe save successfully!")
    print("Dataframe save successfully!")

    return df_crawling

if __name__ == "__main__":
    crawling()