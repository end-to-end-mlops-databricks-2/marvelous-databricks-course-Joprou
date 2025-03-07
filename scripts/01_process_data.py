"""Script to process data."""

import argparse

from loguru import logger
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig
from hotel_reservations.data_processor import DataProcessor, generate_synthetic_data

spark = (
    SparkSession.builder.config("spark.sql.session.timeZone", "UTC").config("spark.driver.memory", "12g").getOrCreate()
)

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", action="store", default=None, type=str, required=True)
parser.add_argument("--env", action="store", default=None, type=str, required=True)
args = parser.parse_args()


config = ProjectConfig.from_yaml(config_path=f"{args.root_path}/files/project_config.yml", env=args.env)
logger.info("Config loaded successfully.")
logger.info(config)

data = spark.read.csv(
    f"/Volumes/{config.catalog_name}/{config.schema_name}/data/Hotel Reservations.csv",
    header=True,
    inferSchema=True,
).toPandas()
logger.info("Data loaded successfully.")

synthetic_data = generate_synthetic_data(data, num_rows=50)

data_processor = DataProcessor(pandas_df=synthetic_data, config=config, spark=spark)

# preprocess data
data_processor.preprocess()

train_set, test_set = data_processor.split_data()

# COMMAND ----------
data_processor.save_to_catalog({"train_set": train_set, "test_set": test_set})
# COMMAND ----------
