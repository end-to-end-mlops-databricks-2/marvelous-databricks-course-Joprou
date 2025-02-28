# Databricks notebook source
# %pip install --force-reinstall hotel_reservations-0.0.1-py3-none-any.whl
# dbutils.library.restartPython()

# COMMAND ----------
# %reload_ext autoreload"
# %autoreload 2
import os

import mlflow
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.serving.feature_serving import FeatureServing
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
feature_table_name = f"{catalog_schema}.hotel_reservations_predictions"
feature_spec_name = f"{catalog_schema}.return_predictions"
ENDPOINT_NAME = "hotel-reservations-feature-serving"

# COMMAND ----------
# prepare dataset
train_set = spark.table(f"{catalog_schema}.train_set").toPandas()
test_set = spark.table(f"{catalog_schema}.test_set").toPandas()
df = pd.concat([train_set, test_set])

# load model
model = mlflow.sklearn.load_model(
    f"models:/{catalog_schema}.hotel_reservations_model_basic@latest-model"
)

# predict
preds_df = df[["Booking_ID"]].copy()
preds_df["predicted_cancelled"] = model.predict(df[config.num_features + config.cat_features])
preds_df = spark.createDataFrame(preds_df)

# create prediction feature table
fe = FeatureEngineeringClient()
fe.create_table(
    name=feature_table_name,
    df=preds_df,
    primary_keys=["Booking_ID"],
    description="Hotel reservations cancellation feature table",
)

spark.sql(
    f"""
ALTER TABLE {feature_table_name}
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
"""
)

# COMMAND ----------
feature_serving = FeatureServing(
    feature_table_name=feature_table_name,
    feature_spec_name=feature_spec_name,
    endpoint_name=ENDPOINT_NAME,
)

# COMMAND ----------
feature_serving.create_online_table()

# COMMAND ----------
feature_serving.create_feature_spec()

# COMMAND ----------
feature_serving.deploy_or_update_serving_endpoint()

# COMMAND ----------
response_status_code, response_text = call_dbr_endpoint(
    dbr_host=os.environ["DBR_HOST"],
    dbr_token=os.environ["DBR_TOKEN"],
    endpoint_name=ENDPOINT_NAME,
    record=[{"Booking_ID": "INN00938"}],
)

print(response_status_code)
print(response_text)
