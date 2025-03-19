# Databricks notebook source
# %pip install --force-reinstall hotel_reservations-0.0.1-py3-none-any.whl
# dbutils.library.restartPython()

# COMMAND ----------


# COMMAND ----------

# %reload_ext autoreload"
# %autoreload 2
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.monitoring import create_or_refresh_monitoring_table

spark = SparkSession.builder.config("spark.sql.session.timeZone", "UTC").getOrCreate()
workspace = WorkspaceClient()
config = ProjectConfig.from_yaml("../project_config.yml")
CATALOG_SCHEMA = f"{config.catalog_name}.{config.schema_name}"

# COMMAND ----------
create_or_refresh_monitoring_table(
    spark=spark,
    workspace=workspace,
    catalog_schema=CATALOG_SCHEMA,
    inf_table_name=f"{CATALOG_SCHEMA}.hotel_reservations_payload",
    monitoring_table_name=f"{CATALOG_SCHEMA}.hotel_reservations_monitoring",
)

# COMMAND ----------
