# Databricks notebook source

import pandas as pd
import logging

from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor
from pyspark.sql import SparkSession


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

data = pd.read_csv("../data/Hotel Reservations.csv")
logging.info("Data loaded successfully")

config = ProjectConfig.from_yaml("../project_config.yml")
logging.info("Config loaded successfully")

# COMMAND ----------

data_processor = DataProcessor(pandas_df=data, config=config, spark=spark)

# preprocess data
data_processor.preprocess()
# COMMAND ----------

train_set, test_set = data_processor.split_data()

# COMMAND ----------
data_processor.save_to_catalog({"train_set": train_set, "test_set": test_set})
# COMMAND ----------
