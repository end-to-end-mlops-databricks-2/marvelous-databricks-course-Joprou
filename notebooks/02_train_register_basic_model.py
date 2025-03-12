# Databricks notebook source
# %pip install hotel_reservations-0.0.1-py3-none-any.whl
# dbutils.library.restartPython()

# COMMAND ----------
# %reload_ext autoreload
# %autoreload 2

import mlflow
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.basic_model import BasicModel

# COMMAND ----------
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
# COMMAND ----------
config = ProjectConfig.from_yaml("../project_config.yml")
spark = SparkSession.builder.config("spark.sql.session.timeZone", "UTC").getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2", "job_run_id": "a1b2"})
# COMMAND ----------
basic_model = BasicModel(config=config, tags=tags, spark=spark)
# COMMAND ----------
basic_model.load_data()
basic_model.prepare_model_pipeline()
# COMMAND ----------
basic_model.train_model()
# COMMAND ----------
basic_model.log_model()

# COMMAND ----------
# Retrieve dataset for the current run
basic_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
basic_model.register_model()

# COMMAND ----------
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()
predictions_df = basic_model.load_latest_model_and_predict(X_test)

# COMMAND ----------
