import argparse

from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.serving.model_serving import ModelServing

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", action="store", default=None, type=str, required=True)
parser.add_argument("--env", action="store", default=None, type=str, required=True)
args = parser.parse_args()

config = ProjectConfig.from_yaml(
    config_path=f"{args.root_path}/files/project_config.yml", env=args.env
)
logger.info("Config loaded successfully.")
CATALOG_SCHEMA = f"{config.catalog_name}.{config.schema_name}"
ENDPOINT_NAME = f"hotel-reservations-model-serving-basic-{args.env}"

spark = SparkSession.builder.config("spark.sql.session.timeZone", "UTC").getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

model_server = ModelServing(
    model_name=f"{CATALOG_SCHEMA}.hotel_reservations_model",
    endpoint_name=ENDPOINT_NAME,
)

model_server.deploy_or_update_serving_endpoint(version=model_version)
