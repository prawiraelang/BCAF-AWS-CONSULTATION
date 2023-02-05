# Pipeline for Training Pipeline
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
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor

from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterInteger,
    ParameterString,
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
    base_job_prefix="UsedCarML",
    processing_instance_type="ml.m5.xlarge",
    processing_instance_count=1,
    training_instance_type="ml.m5.xlarge",
    training_instance_count=1,
    transform_instances_type="ml.m5.large"
):
    sagemaker_session = get_session(region, default_bucket)
   
    if default_bucket is None:
        default_bucket = "sample-bucket-11" # Gunakan bucket yang ada di Singapore
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)
    
    # Parameters for pipeline execution
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    transform_instances_type = ParameterString(name="TransformInstanceType", default_value="ml.m5.large")
    
    # Ganti path sesuai lokasi file di direktori S3 dan pastikan file sudah memiliki format CSV
    input_data_lelang = ParameterString(
        name="InputDataLelangURI",
        default_value=f"s3://bcafbucket/training/raw/lelang/used-car-price-prediction-cleaned.csv",
    )
    input_data_crawling = ParameterString(
        name="InputDataCrawlingURI",
        default_value=f"s3://bcafbucket/training/raw/crawling/used-car-price-prediction-cleaned.csv",
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

    model_path = f"s3://bcafbucket/model" # Path tempat menyimpan model di bucket region Jakarta 
    unique_key = strftime("%Y%m%d-%H:%M:%S", gmtime())
    model_name = f"xgboost-model{unique_key}"
    training_job_name = f"xgboost-training-job{unique_key}"
    
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.5-1",
        py_version="py3",
        instance_type=training_instance_type,
    )

    xgb = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        output_path=model_path,
        sagemaker_session=pipeline_session,
        role=role,
        base_job_name="xgboost-training-job-sample"
    )

    # Silakan sesuaikan hyperparameter sesuai dengan hyperparameter terbaik dari percobaan yang pernah dilakukan
    xgb.set_hyperparameters(
        objective="reg:squarederror",
        num_round=10,
        max_depth=6,
        eta=0.3,
        gamma=4,
        min_child_weight=6,
        subsample=0.9
    )

    step_args = xgb.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri, content_type="text/csv"),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri, content_type="text/csv",),
        },
    )

    step_train = TrainingStep(
        name="TrainUsedCarMLModel",
        step_args=step_args
    )

    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
        name=model_name
    )

    step_args = model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium"
    )

    step_create_model = ModelStep(
        name="UsedCarMLCreateModel",
        step_args=step_args,
    )
    
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/Eval",
        sagemaker_session=pipeline_session,
        role=role,
    )
    
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
    )
    evaluation_report = PropertyFile(
        name="MLEvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    step_eval = ProcessingStep(
        name="EvaluateUsedCarMLModel",
        step_args=step_args,
        property_files=[evaluation_report]
    )

    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=transform_instances_type,
        model_package_group_name=model_package_group_name,
    )

    step_register = ModelStep(
        name="RegisterUsedCarMLModel",
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
        steps=[step_process, step_train, step_create_model, step_eval],
        sagemaker_session=pipeline_session,
    )
    return pipeline