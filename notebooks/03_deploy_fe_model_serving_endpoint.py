# Databricks notebook source
# %pip install --force-reinstall hotel_reservations-0.0.1-py3-none-any.whl
# dbutils.library.restartPython()

# COMMAND ----------
import os
import time

import mlflow
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.serving.fe_model_serving import FeatureLookupServing
from hotel_reservations.utils import call_dbr_endpoint

# COMMAND ----------
spark = SparkSession.builder.config("spark.sql.session.timeZone", "UTC").getOrCreate()
dbutils = DBUtils(spark)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
# COMMAND ----------
os.environ["DBR_TOKEN"] = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
)
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")


# COMMAND ----------
config = ProjectConfig.from_yaml("../project_config.yml")
catalog_schema = f"{config.catalog_name}.{config.schema_name}"
ENDPOINT_NAME = "hotel-reservations-model-serving-fe"

# COMMAND ----------
feature_lookup_server = FeatureLookupServing(
    model_name=f"{catalog_schema}.hotel_reservations_model_fe",
    endpoint_name=ENDPOINT_NAME,
    feature_table_name=f"{catalog_schema}.hotel_features",
)


# COMMAND ----------
feature_lookup_server.create_online_table()

# COMMAND ----------
feature_lookup_server.deploy_or_update_serving_endpoint()


# COMMAND ----------
train_set = spark.table(f"{catalog_schema}.train_set").limit(20)

# Drop feature look-up columns and target column
sample = train_set.drop(
    "no_of_previous_cancellations", "no_of_special_requests", "update_timestamp_utc", config.target
)
sample_records = sample.toPandas().to_dict(orient="records")
dataframe_records = [[record] for record in sample_records]


# COMMAND ----------
# Call endpoint using test record
status_code, response_text = call_dbr_endpoint(
    dbr_host=os.environ["DBR_HOST"],
    dbr_token=os.environ["DBR_TOKEN"],
    endpoint_name=ENDPOINT_NAME,
    record=dataframe_records[0],
)
print(f"Status code: {status_code}")
print(f"Response: {response_text}")


for i, record in enumerate(dataframe_records):
    status_code, response_text = call_dbr_endpoint(
        dbr_host=os.environ["DBR_HOST"],
        dbr_token=os.environ["DBR_TOKEN"],
        endpoint_name=ENDPOINT_NAME,
        record=record,
    )
    print(f"Response for record {i}: {response_text}")
    time.sleep(0.2)
