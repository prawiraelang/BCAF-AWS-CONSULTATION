import boto3
from logger_util import logger

'''
Edit below section only according to your needs!
'''
MODEL_TYPES = ['toyota', 'daihatsu']
'''
Edit above section only according to your needs!
'''

pipeline_name = {model_type: f'GLAIR-BCAF-Consultation-BatchTransform-{model_type.capitalize()}' for model_type in MODEL_TYPES}

# Set up the SageMaker and S3 client with the appropriate region
sagemaker_virginia = boto3.client('sagemaker', region_name='us-east-1')
s3_singapore = boto3.client("s3", region_name="ap-southeast-1")

def run_pipeline(pipeline_name, s3_path, model_type):
    def get_latest_file(bucket_name, prefix_name):
        s3_uri_response = s3_singapore.list_objects_v2(Bucket=bucket_name, Prefix=prefix_name)
        csv_keys = [obj for obj in s3_uri_response.get("Contents", []) if obj["Key"].endswith(".csv")]
        latest_csv_key = sorted(csv_keys, key=lambda x: x["LastModified"], reverse=True)[0]["Key"]
        
        return f"s3://{bucket_name}/{latest_csv_key}"
    
    def get_latest_model():
        model_response = sagemaker_virginia.list_models(
            SortBy='CreationTime',
            SortOrder='Descending',
            NameContains="Toyota"
        )
    
        s3_uri_response = sagemaker_virginia.describe_model(
            ModelName=model_response['Models'][0]['ModelName']
        )
    
        return s3_uri_response['PrimaryContainer']['ModelDataUrl'], model_response['Models'][0]['ModelName']

    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    prefix = key.split("/", 1)[0]
    
    s3_uri_lelang = s3_path
    s3_uri_crawling = get_latest_file(
        bucket,
        f'{prefix}/training/{model_type}/crawling'
    )
    
    model_name = get_latest_model()[1]
    
    logger.info(f'The latest {model_type.capitalize()} file for lelang data is located at "{s3_uri_lelang}"')
    logger.info(f'The latest {model_type.capitalize()} file for crawling data is located at "{s3_uri_crawling}"')
    logger.info(f'The latest {model_type.capitalize()} model is located at "{model_name}"')

    # Start the pipeline execution with defined parameters
    execution_response = sagemaker_virginia.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineParameters=[
            { 
                 "Name": "ProcessingInstanceType",
                 "Value": "ml.m5.large"
            },
            { 
                 "Name": "ProcessingInstanceCount",
                 "Value": "1"
            },
            { 
                 "Name": "TrainingInstanceType",
                 "Value": "ml.m5.large"
            },
            { 
                 "Name": "TransformInstanceType",
                 "Value": "ml.m5.large"
            },
            { 
                 "Name": "TransformInstanceCount",
                 "Value": "1"
            },
            { 
                 "Name": "InputDataLelangURI",
                 "Value": s3_uri_lelang
            },
            { 
                 "Name": "InputDataCrawlingURI",
                 "Value": s3_uri_crawling
            },
            { 
                 "Name": "ModelName",
                 "Value": model_name
            }
        ]
    )

    # Print the pipeline execution ARN to the logs
    logger.info(
        f'Pipeline execution started on "{execution_response["PipelineExecutionArn"]}"'
    )
    
def lambda_handler(event, context):
    try:
        # Get the S3 path that triggered this Lambda function
        s3_paths = []
        for record in event["Records"]:
            bucket = record["s3"]["bucket"]["name"]
            key = record["s3"]["object"]["key"]
            s3_paths.append(f"s3://{bucket}/{key}")
            
        # Print the S3 path to the logs
        logger.info(
            f'S3 path that triggered this Lambda function is "{",".join(s3_paths)}"'
        )
        
        for s3_path in s3_paths:
            for model_type in MODEL_TYPES:
                if model_type + SUFFIX_DIR in s3_path:
                    run_pipeline(pipeline_name[model_type], s3_path, model_type)
        
    except Exception as e:
        logger.error(e)
        raise e
        
    return {
        'statusCode': 200,
        'body': 'Pipeline execution has started!'
    }