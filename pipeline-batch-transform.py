import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput, CreateModelInput, TransformInput
from sagemaker.model import Model
from sagemaker.transformer import Transformer

from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor
)
from sagemaker.sklearn.processing import SKLearnProcessor

from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterInteger,
    ParameterString
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CreateModelStep,
    TransformStep
)

from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from time import gmtime, strftime

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_session(region, default_bucket):
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
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="UsedCarMLPackageGroup",
    pipeline_name="UsedCarMLPipeline",
    base_job_prefix="UsedCarMLPrediction",
    processing_instance_type="ml.m5.xlarge",
    processing_instance_count=1,
    training_instance_type="ml.m5.xlarge",
    training_instance_count=1,
    transform_instances_type="ml.m5.large"
):
    sagemaker_session = get_session(region, default_bucket)
   
    if default_bucket is None:
        default_bucket = "glair-exploration-sagemaker-bucket"
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)
    
    # Parameters for pipeline execution
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    transform_instances_type = ParameterString(name="TransformInstanceType", default_value="ml.m5.large")
    
    input_data_lelang = ParameterString(
        name="InputDataLelangURI",
        default_value=f"s3://glair-exploration-sagemaker-bucket/glair-bcaf-consultation/training/raw/used-car-price-prediction-cleaned.csv",
    )
    input_data_crawling = ParameterString(
        name="InputDataCrawlingURI",
        default_value=f"s3://glair-exploration-sagemaker-bucket/glair-bcaf-consultation/training/raw/used-car-price-prediction-cleaned.csv",
    )

    # Processing step
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/Preprocess",
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data-lelang", input_data_lelang,
                   "--input-data-crawling", input_data_crawling]
    )

    step_process = ProcessingStep(
        name="PreprocessUsedCarMLData",
        step_args=step_args,        
    )

    model_path = f"s3://{default_bucket}/glair-bcaf-consultation/model"
    unique_key = strftime("%Y%m%d-%H:%M:%S", gmtime())
    model_name = f"xgboost-bcaf-model{unique_key}"
    transform_job_name = f"xgboost-transform-job{unique_key}"
    
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.5-1",
        py_version="py3",
        instance_type=training_instance_type,
    )

    model = Model(
        image_uri=image_uri,
        model_data="s3://glair-exploration-sagemaker-bucket/glair-bcaf-consultation/model/pipelines-25kejldvtw4g-TrainUsedCarMLModel-TqbxH2HVo8/output/model.tar.gz",
        sagemaker_session=pipeline_session,
        role=role,
        name=model_name
    )

    step_args = model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
    )

    step_create_model = ModelStep(
        name="UsedCarMLCreateModel",
        step_args=step_args,
    )

    # Batch transform step
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        accept="text/csv",
        assemble_with="Line",
        output_path=f"s3://{default_bucket}//glair-bcaf-consultation/batch_transform/output",
        sagemaker_session=pipeline_session,
    )

    transform_inputs = TransformInput(
        data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
    )

    step_args = transformer.transform(
        data=transform_inputs.data,
        job_name=transform_job_name,
        input_filter="$[1:]",
        join_source="Input",
        output_filter="$[0,1]",
        content_type="text/csv",
        split_type="Line",
    )

    step_transform = TransformStep(
        name="UsedCarPredictionTransform",
        step_args=step_args,
    )
    

    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            training_instance_count,
            transform_instances_type,
            input_data_lelang,
            input_data_crawling
            
        ],
        steps=[step_process, step_create_model, step_transform ],
        sagemaker_session=pipeline_session,
    )
    return pipeline
