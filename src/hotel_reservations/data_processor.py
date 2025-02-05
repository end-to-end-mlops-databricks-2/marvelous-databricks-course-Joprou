"""Data processing module."""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from hotel_reservations.config import ProjectConfig


class DataProcessor:
    """Data processing class."""

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession):
        self.df = pandas_df
        self.config = config
        self.spark = spark

    def preprocess(self):
        """Preprocess the data."""
        cat_cols = self.config.cat_features
        num_cols = self.config.num_features

        # Drop columns
        self.df.drop(self.config.drop_cols, axis=1, inplace=True)

        self.df[cat_cols] = self.df[cat_cols].astype("category")
        self.df[num_cols] = self.df[num_cols].apply(pd.to_numeric, errors="coerce")

    def split_data(self, test_size: int = 0.2, random_state=0, **kwargs):
        """Split the data into train and test sets."""

        return train_test_split(self.df, test_size=test_size, random_state=random_state, **kwargs)

    def save_to_catalog(self, table_config: dict[str, pd.DataFrame]):
        """Save df into Databricks tables.

        Args:
            table_config (dict[str, pd.DataFrame]): Set of dataframes to be pushed to \
                Databricks catalog, where the key is the table name and value is the \
                dataframe.
        """

        for table_name, df in table_config.items():
            df_with_timestamp = self.spark.createDataFrame(df).withColumn(
                "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
            )

            df_with_timestamp.write.mode("append").saveAsTable(
                f"{self.config.catalog_name}.{self.config.schema_name}.{table_name}"
            )
