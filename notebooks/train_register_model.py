# Databricks notebook source

import os
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
tags = Tags(**{"git_sha": "abcd12345", "branch": "week2"})
# COMMAND ----------
basic_model = BasicModel(config=config, tags=tags, spark=spark)
# COMMAND ----------
basic_model.load_data()
basic_model.prepare_model_pipeline()
# COMMAND ----------
basic_model.train_model()
basic_model.evaluate_model()
# COMMAND ----------
basic_model.log_model()
# COMMAND ----------
