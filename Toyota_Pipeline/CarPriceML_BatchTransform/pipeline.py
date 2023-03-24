# Pipeline for Batch Transform With Constant
import os
import boto3
import sagemaker
import sagemaker.session

from sagemaker.inputs import TransformInput
from sagemaker.transformer import Transformer
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TransformStep, CacheConfig
from sagemaker.workflow.pipeline_context import PipelineSession
from time import gmtime, strftime

'''
Edit below section only according to your needs!
'''
DEFAULT_BUCKET = "glair-exploration-bcaf-consultation"
PREFIX_MODEL = "glair-bcaf-consultation-output/model"
PREFIX_EVALUATION = "glair-bcaf-consultation-output/evaluation"
PREFIX_BATCH_TRANSFORM = "glair-bcaf-consultation-output/batch-transform"
MODEL_TYPE = "toyota"
'''
Edit above section only according to your needs!
'''

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket
    )


def get_pipeline_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket
    )

def get_pipeline(
    region=None,
    role=None,
    default_bucket=None,
    pipeline_name=None,
    processing_instance_type=None,
    processing_instance_count=None,
    training_instance_type=None,
    transform_instances_type=None,
    transform_instances_count=None
):
    
    if region is None:
        region = boto3.Session().region_name
        
    if default_bucket is None:
        default_bucket = DEFAULT_BUCKET # To save and run the pipeline
    
    sagemaker_session = get_sagemaker_session(region, default_bucket)
    pipeline_session = get_pipeline_session(region, default_bucket)
    
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    
    # Parameters for pipeline execution
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.large")
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.large")
    transform_instances_type = ParameterString(name="TransformInstanceType", default_value="ml.m5.large")
    transform_instances_count = ParameterInteger(name="TransformInstanceCount", default_value=1)
    
    input_data_lelang = ParameterString(name="InputDataLelangURI")
    input_data_crawling = ParameterString(name="InputDataCrawlingURI")
    
    model_name = ParameterString(name="ModelName")

    # Cache Pipeline steps to reduce execution time on subsequent executions
    cache_config = CacheConfig(enable_caching=True, expire_after="7d")
    
    tags_dict = [
        {
            'Key': 'Platform',
            'Value': 'consulting'
        },
        {
            'Key': 'Tenant',
            'Value': 'bca-finance'
        },
        {
            'Key': 'Environment',
            'Value': 'development'
        },
    ]
    
    # Processing step
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        sagemaker_session=pipeline_session,
        role=role,
        tags=tags_dict
    )

    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="predict", source="/opt/ml/processing/predict")
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data-lelang", input_data_lelang,
                   "--input-data-crawling", input_data_crawling]
    )

    step_preprocess = ProcessingStep(
        name=f"{MODEL_TYPE.capitalize()}-CarPriceML-Preprocess",
        step_args=step_args      
    )

    # unique_key = strftime("%Y%m%d-%H:%M:%S", gmtime())
    unique_key = strftime("%Y%m%d", gmtime())

    # Batch transform step
    batch_transform_path = f"s3://{DEFAULT_BUCKET}/{PREFIX_BATCH_TRANSFORM}/{MODEL_TYPE}/raw/{unique_key}"
    transformer = Transformer(
        model_name=model_name,
        instance_type=transform_instances_type,
        instance_count=transform_instances_count,
        accept="text/csv",
        assemble_with="Line",
        output_path=batch_transform_path,
        sagemaker_session=pipeline_session,
        tags=tags_dict
    )

    transform_inputs = TransformInput(
        data=step_preprocess.properties.ProcessingOutputConfig.Outputs["predict"].S3Output.S3Uri,
    )

    # If you want to filter the output of batch transform or join it with the input source, use this block of code
    # step_args = transformer.transform(
    #     data=transform_inputs.data,
    #     input_filter="$[0:]",
    #     join_source="Input",
    #     output_filter="$[0,-1]",
    #     content_type="text/csv",
    #     split_type="Line"
    # )
    
    # If you only want the label column of batch transform, use this block of code
    step_args = transformer.transform(
        data=transform_inputs.data,
        input_filter="$[0:]",
        content_type="text/csv",
        split_type="Line"
    )

    step_transform = TransformStep(
        name=f"{MODEL_TYPE.capitalize()}-CarPriceML-BatchTransform",
        step_args=step_args,
        cache_config=cache_config
    )

    # Postprocessing step
    step_args = sklearn_processor.run(
        code=os.path.join(BASE_DIR, "postprocess.py"),
        arguments=["--input-data-lelang", input_data_lelang,
                   "--input-data-crawling", input_data_crawling,
                   "--input-batch-transform", step_transform.properties.TransformOutput.S3OutputPath,
                   "--model-type", MODEL_TYPE,
                   "--prefix-batch-transform", PREFIX_BATCH_TRANSFORM]
    )

    step_postprocess = ProcessingStep(
        name=f"{MODEL_TYPE.capitalize()}-CarPriceML-Postprocess",
        step_args=step_args
    )
    
    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            transform_instances_type,
            transform_instances_count,
            input_data_lelang,
            input_data_crawling,
            model_name
        ],
        steps=[step_preprocess, step_transform, step_postprocess],
        sagemaker_session=pipeline_session,
    )
    
    return pipeline