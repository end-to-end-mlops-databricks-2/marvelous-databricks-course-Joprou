# Databricks notebook source

%load_ext autoreload
%autoreload 2

import pandas as pd
import logging

from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# COMMAND ----------

data = pd.read_csv("../data/Hotel Reservations.csv")
logging.info("Data loaded successfully")

config = ProjectConfig.from_yaml("../project_config.yml")
logging.info("Config loaded successfully")

# COMMAND ----------

data_processor = DataProcessor(pandas_df=data, config=config)

# preprocess data
data_processor.preprocess()
# COMMAND ----------

X_train, X_test, y_train, y_test = data_processor.split_data()
# COMMAND ----------
