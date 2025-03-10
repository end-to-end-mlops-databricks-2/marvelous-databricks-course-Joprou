import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.feature_lookup_model import FeatureLookUpModel

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", action="store", default=None, type=str, required=True)
parser.add_argument("--env", action="store", default=None, type=str, required=True)
parser.add_argument("--git_sha", action="store", default=None, type=str, required=True)
parser.add_argument("--job_run_id", action="store", default=None, type=str, required=True)
parser.add_argument("--branch", action="store", default=None, type=str, required=True)
args = parser.parse_args()


spark = SparkSession.builder.config("spark.sql.session.timeZone", "UTC").getOrCreate()
dbutils = DBUtils(spark)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)
config = ProjectConfig.from_yaml(config_path=f"{args.root_path}/files/project_config.yml", env=args.env)

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
logger.info("Model initialized.")

# Create feature table
fe_model.create_or_update_feature_table()

# load data
fe_model.load_data()
logger.info("Data loaded.")

# Execute feature engineering
fe_model.feature_engineering()

# Train model
fe_model.train()

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").drop(
    "no_of_previous_cancellations", "no_of_special_requests"
)

if fe_model.is_model_improves(test_set):
    logger.info("Model improved. Registering new model.")
    latest_version = fe_model.register_model()
    logger.info(f"Model registered with version {latest_version}.")
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)
else:
    logger.info("Model did not improve. Keeping the current model.")
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
