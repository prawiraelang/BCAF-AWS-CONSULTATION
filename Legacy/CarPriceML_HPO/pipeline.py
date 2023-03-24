# Pipeline for HPO
import os
import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TuningStep, CacheConfig
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
from time import gmtime, strftime

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
    training_instance_count=None
):
    if region is None:
        region = boto3.Session().region_name
        
    if default_bucket is None:
        default_bucket = "glair-exploration-sagemaker-bucket" # To save and run the pipeline
    
    sagemaker_session = get_sagemaker_session(region, default_bucket)
    pipeline_session = get_pipeline_session(region, default_bucket)
    
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
        
    # Parameters for pipeline execution
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)

    # Cache Pipeline steps to reduce execution time on subsequent executions
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")
    
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
    
    s3_singapore = boto3.client("s3", region_name="ap-southeast-1")

    def get_latest_file(bucket_name, prefix_name):
        s3_uri_response = s3_singapore.list_objects_v2(Bucket=bucket_name, Prefix=prefix_name)
        latest_key = sorted(s3_uri_response.get("Contents", []), key=lambda x: x["LastModified"], reverse=True)[0]["Key"]
    
        return f"s3://{bucket_name}/{latest_key}"
    
    s3_uri_lelang = get_latest_file(
        "glair-exploration-sagemaker-s3-bucket-singapore",
        "glair-bcaf-consultation-input/training/lelang"
    )

    s3_uri_crawling = get_latest_file(
        "glair-exploration-sagemaker-s3-bucket-singapore",
        "glair-bcaf-consultation-input/training/crawling"
    )

    input_data_lelang = ParameterString(
        name="InputDataLelangURI",
        default_value=s3_uri_lelang
    )
    input_data_crawling = ParameterString(
        name="InputDataCrawlingURI",
        default_value=s3_uri_crawling
    )

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
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data-lelang", input_data_lelang,
                   "--input-data-crawling", input_data_crawling]
    )

    step_preprocess = ProcessingStep(
        name="CarPriceML-Preprocess",
        step_args=step_args      
    )

    # unique_key = strftime("%Y%m%d-%H:%M:%S", gmtime())
    unique_key = strftime("%Y%m%d", gmtime())
    
    # HPO step
    model_path = f"s3://glair-exploration-sagemaker-bucket/glair-bcaf-consultation-output/model/{unique_key}"
    model_prefix_name = f"glair-bcaf-consultation-output/model/{unique_key}"
    
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.5-1",
        py_version="py3",
        instance_type=training_instance_type
    )

    xgb = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        output_path=model_path,
        sagemaker_session=pipeline_session,
        role=role
    )

    xgb.set_hyperparameters(
        objective="reg:squarederror",
        verbosity=0
    )

    objective_metric_name = "validation:rmse"

    hyperparameter_ranges = {
        "max_depth": IntegerParameter(3, 10, scaling_type="Auto"),
        "subsample": ContinuousParameter(0.5, 1.0, scaling_type="Auto"),
        "colsample_bytree": ContinuousParameter(0.5, 1.0, scaling_type="Auto"),
        "num_round": IntegerParameter(50, 500, scaling_type="Auto"),
        "eta": ContinuousParameter(0.01, 0.3, scaling_type="Auto"),
        "min_child_weight": IntegerParameter(1, 10, scaling_type="Auto"),
        "gamma": IntegerParameter(1, 5, scaling_type="Auto")
    }

    tuner_log = HyperparameterTuner(
        xgb,
        objective_metric_name,
        hyperparameter_ranges,
        max_jobs=50,
        max_parallel_jobs=3,
        strategy="Bayesian",
        objective_type="Minimize",
        tags=tags
    )

    hpo_args = tuner_log.fit(
        inputs={
            "train": TrainingInput(
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri, content_type="csv"),
            "validation": TrainingInput(
            s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri, content_type="csv")
        }
    )

    step_tuning = TuningStep(
        name="CarPriceML-HPO",
        step_args=hpo_args,
        cache_config=cache_config
    )

    # Create Model step
    model = Model(
        image_uri=image_uri,
        model_data=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=default_bucket, prefix=model_prefix_name),
        sagemaker_session=pipeline_session,
        role=role
    )

    step_args = model.create(
        instance_type="ml.m5.xlarge",
        accelerator_type="ml.eia1.medium",
        tags=tags_dict
    )

    step_create_model = ModelStep(
        name="CarPriceML",
        step_args=step_args
    )
    
    # Evaluation step
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        sagemaker_session=pipeline_session,
        role=role,
        tags=tags_dict
    )
    
    step_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=default_bucket, 
                                                        prefix=model_prefix_name),
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
        ],
        code=os.path.join(BASE_DIR, "evaluate.py")
    )
    
    step_eval = ProcessingStep(
        name="CarPriceML-Evaluate",
        step_args=step_args
    )

    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            training_instance_count,
            input_data_lelang,
            input_data_crawling 
        ],
        steps=[step_preprocess, step_tuning, step_create_model, step_eval],
        sagemaker_session=pipeline_session,
    )
    
    return pipeline