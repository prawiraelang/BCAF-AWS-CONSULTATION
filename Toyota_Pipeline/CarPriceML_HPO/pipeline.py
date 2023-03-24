# Pipeline for HPO With Constant
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

'''
Edit below section only according to your needs!
'''
DEFAULT_BUCKET = "glair-exploration-bcaf-consultation"
PREFIX_PREPROCESS = "glair-bcaf-consultation-output/training"
PREFIX_MODEL = "glair-bcaf-consultation-output/model"
PREFIX_EVALUATION = "glair-bcaf-consultation-output/evaluation"
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
    training_instance_count=None
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
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)

    input_data_lelang = ParameterString(name="InputDataLelangURI")
    input_data_crawling = ParameterString(name="InputDataCrawlingURI")

    # Cache Pipeline steps to reduce execution time on subsequent executions
    cache_config = CacheConfig(enable_caching=True, expire_after="90d")
    
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
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        arguments=["--input-data-lelang", input_data_lelang,
                   "--input-data-crawling", input_data_crawling,
                   "--default-bucket", DEFAULT_BUCKET,
                   "--model-type", MODEL_TYPE,
                   "--prefix-preprocess", PREFIX_PREPROCESS]
    )

    step_preprocess = ProcessingStep(
        name=f"{MODEL_TYPE.capitalize()}-CarPriceML-Preprocess",
        step_args=step_args      
    )

    # unique_key = strftime("%Y%m%d-%H:%M:%S", gmtime())
    unique_key = strftime("%Y%m%d", gmtime())
    
    # HPO step
    model_path = f"s3://{DEFAULT_BUCKET}/{PREFIX_MODEL}/{MODEL_TYPE}/{unique_key}"
    model_prefix_name = f"{PREFIX_MODEL}/{MODEL_TYPE}/{unique_key}"
    
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
        max_jobs=10,
        max_parallel_jobs=3,
        strategy="Bayesian",
        objective_type="Minimize",
        tags=tags_dict
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
        name=f"{MODEL_TYPE.capitalize()}-CarPriceML-HPO",
        step_args=hpo_args,
        cache_config=cache_config
    )

    # Create Model step
    model = Model(
        image_uri=image_uri,
        model_data=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=DEFAULT_BUCKET, prefix=model_prefix_name),
        sagemaker_session=pipeline_session,
        role=role
    )

    step_args = model.create(
        instance_type="ml.m5.large",
        accelerator_type="ml.eia1.medium",
        tags=tags_dict
    )

    step_create_model = ModelStep(
        name=f"{MODEL_TYPE.capitalize()}-CarPriceML",
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
        code=os.path.join(BASE_DIR, "evaluate.py"),
        arguments=["--default-bucket", DEFAULT_BUCKET,
                   "--model-type", MODEL_TYPE,
                   "--prefix-evaluation", PREFIX_EVALUATION]
    )
    
    step_eval = ProcessingStep(
        name=f"{MODEL_TYPE.capitalize()}-CarPriceML-Evaluate",
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