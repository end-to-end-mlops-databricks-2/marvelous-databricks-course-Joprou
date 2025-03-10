import argparse

from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.serving.fe_model_serving import FeatureLookupServing

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", action="store", default=None, type=str, required=True)
parser.add_argument("--env", action="store", default=None, type=str, required=True)
args = parser.parse_args()

config = ProjectConfig.from_yaml(config_path=f"{args.root_path}/files/project_config.yml", env=args.env)
logger.info("Config loaded successfully.")
catalog_schema = f"{config.catalog_name}.{config.schema_name}"
ENDPOINT_NAME = f"hotel-reservations-model-serving-fe-{args.env}"

spark = SparkSession.builder.config("spark.sql.session.timeZone", "UTC").getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

feature_lookup_server = FeatureLookupServing(
    model_name=f"{catalog_schema}.hotel_reservations_model_fe",
    endpoint_name=ENDPOINT_NAME,
    feature_table_name=f"{catalog_schema}.hotel_features",
)

# feature_lookup_server.create_online_table()
feature_lookup_server.update_online_table(pipeline_id=config.pipeline_id)

feature_lookup_server.deploy_or_update_serving_endpoint(version=model_version)
