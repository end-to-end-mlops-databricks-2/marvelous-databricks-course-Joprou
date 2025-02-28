# Databricks notebook source
# %pip install --force-reinstall hotel_reservations-0.0.1-py3-none-any.whl
# dbutils.library.restartPython()

# COMMAND ----------
# %reload_ext autoreload
# %autoreload 2

import mlflow
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.feature_lookup_model import FeatureLookUpModel

# COMMAND ----------
spark = SparkSession.builder.config("spark.sql.session.timeZone", "UTC").getOrCreate()
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

tags_dict = {"git_sha": "abcd12345", "branch": "week2"}
tags = Tags(**tags_dict)
config = ProjectConfig.from_yaml("../project_config.yml")

# COMMAND ----------
# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------
# Create feature table
fe_model.create_feature_table()

# COMMAND ----------
# load data
fe_model.load_data()

# COMMAND ----------
# Execute feature engineering
fe_model.feature_engineering()

# COMMAND ----------
# Train model
fe_model.train()

# COMMAND ----------
fe_model.register_model()

# COMMAND ----------
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# Drop feature look-up columns and target column
x_test = test_set.drop("no_of_previous_cancellations", "no_of_special_requests", config.target)
predictions = fe_model.load_latest_model_and_predict(x_test, result_type="string")
predictions_df = predictions.toPandas()
