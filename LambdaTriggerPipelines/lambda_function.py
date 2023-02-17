import boto3
from logger_util import logger

def lambda_handler(event, context):
    try:
        # Get the S3 path that triggered this Lambda function
        s3_path = []
        for record in event["Records"]:
            bucket = record["s3"]["bucket"]["name"]
            key = record["s3"]["object"]["key"]
            
            s3_path.append(f"s3://{bucket}/{key}")
            
        # Print the S3 path to the logs
        logger.info(
            f'S3 path that triggered this Lambda function is {",".join(s3_path)}'
        )
    
        # Set up the SageMaker client with the appropriate region
        sagemaker = boto3.client('sagemaker', region_name='region-for-input-data')
        
        # Define the name of the pipeline you want to start
        pipeline_name = "<pipeline_name>"
    
        # Start the pipeline execution with defined parameters
        execution_response = sagemaker.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineParameters=[
                { 
                     "Name": "ProcessingInstanceType",
                     "Value": "ml.m5.xlarge"
                },
                { 
                     "Name": "ProcessingInstanceCount",
                     "Value": "1"
                },
                { 
                     "Name": "TrainingInstanceType",
                     "Value": "ml.m5.xlarge"
                },
                { 
                     "Name": "TransformInstanceType",
                     "Value": "ml.m5.xlarge"
                },
                { 
                     "Name": "TransformInstanceCount",
                     "Value": "1"
                }
            ]
            
        )
    
        # Print the pipeline execution ARN to the logs
        logger.info(
            f'Pipeline execution started on {execution_response["PipelineExecutionArn"]}'
        )
        
    except Exception as e:
        logger.error(e)
        raise e
        
    return {
        'statusCode': 200,
        'body': 'Pipeline execution has started!'
    }