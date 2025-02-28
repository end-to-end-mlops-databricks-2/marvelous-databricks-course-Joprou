# Databricks notebook source
# %pip install --force-reinstall hotel_reservations-0.0.1-py3-none-any.whl
# dbutils.library.restartPython()

# COMMAND ----------
# %reload_ext autoreload
# %autoreload 2
import os
import time

import mlflow
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.serving.model_serving import ModelServing
from hotel_reservations.utils import call_dbr_endpoint

# COMMAND ----------
spark = SparkSession.builder.config("spark.sql.session.timeZone", "UTC").getOrCreate()
dbutils = DBUtils(spark)
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
# COMMAND ----------
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")
# COMMAND ----------
# Loading config
config = ProjectConfig.from_yaml("../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
ENDPOINT_NAME = "hotel_reservations"
# COMMAND ----------
# Instantiating ModelServing class
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.hotel_reservations_model_basic",
    endpoint_name=ENDPOINT_NAME,
)

# COMMAND ----------
# Deploying the serving endpoint
model_serving.deploy_or_update_serving_endpoint()

# COMMAND ----------
required_cols = [
    "no_of_adults",
    "no_of_children",
    "no_of_weekend_nights",
    "no_of_week_nights",
    "type_of_meal_plan",
    "required_car_parking_space",
    "room_type_reserved",
    "lead_time",
    "arrival_month",
    "arrival_date",
    "market_segment_type",
    "repeated_guest",
    "no_of_previous_cancellations",
    "no_of_previous_bookings_not_canceled",
    "avg_price_per_room",
    "no_of_special_requests",
    "booking_status",
]

# COMMAND ----------
# Sample records from test set
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
sampled_records = test_set[required_cols].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]
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


for i in range(len(dataframe_records)):
    status_code, response_text = call_dbr_endpoint(
        dbr_host=os.environ["DBR_HOST"],
        dbr_token=os.environ["DBR_TOKEN"],
        endpoint_name=ENDPOINT_NAME,
        record=dataframe_records[i],
    )
    print(f"Response: {response_text}")
    time.sleep(0.2)
