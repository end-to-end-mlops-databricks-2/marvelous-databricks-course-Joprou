import argparse

from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.monitoring import create_or_refresh_monitoring_table

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", action="store", default=None, type=str, required=True)
parser.add_argument("--env", action="store", default=None, type=str, required=True)
args = parser.parse_args()

config = ProjectConfig.from_yaml(config_path=f"{args.root_path}/files/project_config.yml", env=args.env)
logger.info("Config loaded successfully.")
CATALOG_SCHEMA = f"{config.catalog_name}.{config.schema_name}"

spark = SparkSession.builder.config("spark.sql.session.timeZone", "UTC").getOrCreate()
workspace = WorkspaceClient()

create_or_refresh_monitoring_table(
    spark=spark,
    workspace=workspace,
    catalog_schema=CATALOG_SCHEMA,
    inf_table_name=f"{CATALOG_SCHEMA}.hotel_reservations_payload",
    monitoring_table_name=f"{CATALOG_SCHEMA}.hotel_reservations_monitoring",
)
