"""Data processing module."""

import random
import numpy as np
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


def generate_unique_ids(
    existing_ids: list[str],
    num_new_ids: int,
) -> np.array:
    """Generate unique IDs based on existing IDs.

    Args:
        existing_ids (list[str]): List of existing IDs.
        num_new_ids (int): Number of new IDs to generate.

    Returns:
        np.array: Array of new IDs.

    """
    new_ids = set()
    while len(new_ids) < num_new_ids:
        new_id = f"INN{random.randint(10000, 999999)}"
        if new_id not in existing_ids and new_id not in new_ids:
            new_ids.add(new_id)

    return np.array(new_ids)


def generate_synthetic_data(df: pd.DataFrame, num_rows: int = 10) -> pd.DataFrame:
    """Generate synthetic data from existing data.

    Args:
        df (pd.DataFrame): Existing data.
        num_rows (int): Number of rows to generate.

    Returns:
        pd.DataFrame: Synthetic data

    """

    date_cols = ["arrival_year", "arrival_month", "arrival_date"]

    output = pd.DataFrame()

    for col in df.columns:
        if col == "Booking_ID":
            output[col] = generate_unique_ids(df[col], num_rows)

        elif col in date_cols:
            # temporarily assign as same value to faciliate
            # checking of column names
            output[col] = df[col]

        elif pd.api.types.is_numeric_dtype(df[col]):
            output[col] = np.random.normal(df[col].mean(), df[col].std(), num_rows)

        elif isinstance(df[col], pd.CategoricalDtype) or pd.api.types.is_object_dtype(df[col]):
            output[col] = np.random.choice(
                df[col].unique(), num_rows, p=df[col].value_counts(normalize=True)
            )
        else:
            output[col] = np.random.choice(df[col].unique(), num_rows)

    output[date_cols] = df[date_cols].sample(num_rows).reset_index(drop=True)

    assert list(output.columns) == list(df.columns), "Column names do not match"

    return output
