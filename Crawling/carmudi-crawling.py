import boto3
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import json
from time import gmtime, strftime

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def crawling():
    s3_client = boto3.client("s3", region_name="ap-southeast-3") # Override region values

    def get_latest_file(bucket_name, prefix_name):
        s3_uri_response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix_name)
        latest_key = sorted(s3_uri_response.get("Contents", []), key=lambda x: x["LastModified"], reverse=True)[0]["Key"]
    
        return f"s3://{bucket_name}/{latest_key}"
    
    s3_uri_lelang = get_latest_file(
        "input-car-price-ml",
        "training/lelang/"
    )
    
    bucket_lelang = s3_uri_lelang.split("/")[2]
    key_lelang = "/".join(s3_uri_lelang.split("/")[3:])
    
    file_object = s3_client.get_object(Bucket=bucket_lelang, Key=key_lelang)
    
    list_lelang = pd.read_csv(file_object['Body'], sep=";")
    
    # extract transmisi dari Type
    def findCarTransmisi(x):
        if "M/T" in x:
            return 'manual'
        elif "A/T" in x:
            return 'automatic'
        elif "MT" in x:
            return 'manual'
        elif "AT" in x:
            return 'automatic'
        else:
            return 'manual'

    list_lelang["transmisi"] = list_lelang["Type"].apply(findCarTransmisi)
    
    def findCarType(x):
        if " G" in x:
            return 'G'
        elif " VELOZ" in x:
            return 'VELOZ'
        elif " E" in x:
            return 'E'
        elif " S" in x:
            return 'S'
        else : 
            return 'E'
        
    list_lelang['type_detail'] = list_lelang['Type'].apply(findCarType)
    
    def findCC(x):
        if "1.3" in x:
            return '1.3'
        elif "1.5" in x:
            return '1.5'
        else:
            return '1.3'
        
    list_lelang["CC"] = list_lelang["Type"].apply(findCC)
    
    def findCarTransmisi(x):
        if "M/T" in x:
            return 'manual'
        elif "A/T" in x:
            return 'automatic'
        elif "MT" in x:
            return 'manual'
        elif "AT" in x:
            return 'automatic'
        else:
            return 'manual'
    
    list_lelang["transmisi"] = list_lelang["Type"].apply(findCarTransmisi)
    
    list_mobil = list_lelang[['type_detail', 'CC', 'tahun', 'transmisi']]
    
    #proses crawling, durasi tergantung internet
    year           = list_mobil['tahun']
    transmisi       = list_mobil['transmisi']
    type_detail     = list_mobil['type_detail']
    cc = list_mobil['CC']
    
    price       = []
    tahun       = []
    CC          = []
    Type        = []
    
    for i in range(20):
        # logger.info(i)
        alamat = f"https://www.carmudi.co.id/en/cars-for-sale/toyota/avanza/year-{year[i]}/indonesia?transmission={transmisi[i]}"
        req = Request(alamat, headers={'User-Agent': 'Mozilla/5.0'})
        html = urlopen(req).read()
        data = BeautifulSoup(html, 'html.parser')
        script = data.find_all('script', {'type': 'application/ld+json'})[0]
    
        # print(json.loads(script.text))
        script_ = json.loads(script.text)
        # print(script_)
    
        rows = script_[0]['itemListElement']
        # print(rows)
    
        # break
        for a in range(len(rows)):
            harga   = rows[a]['item']['offers']['price']
    
            # year = name[0:4]
            price.append(harga)
            tahun.append(year[i])
            CC.append(cc[i])
            Type.append(type_detail[i])
    
    
    df_hasil = pd.DataFrame({'price': price,'tahun':tahun, 'CC' :CC, 'Type' : Type}, columns=['price', 'tahun','CC','Type'])
    
    from time import gmtime, strftime
    from io import StringIO
    
    # unique_key = strftime("%Y%m%d-%H:%M:%S", gmtime())
    unique_key = strftime("%Y%m%d", gmtime())
    csv_buffer = StringIO()
    
    df_hasil.to_csv(csv_buffer, index=False)
    
    s3_jakarta = boto3.resource("s3", region_name="ap-southeast-3")
    bucket = s3_jakarta.Bucket("input-car-price-ml")
    
    bucket.put_object(Key=f"training/crawling/{unique_key}/data-crawling-{unique_key}.csv", Body=csv_buffer.getvalue())
    
    return "Crawling success!"
    
crawling()