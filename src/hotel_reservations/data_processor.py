"""Data processing module."""

import pandas as pd
from sklearn.model_selection import train_test_split
from hotel_reservations.config import ProjectConfig


class DataProcessor:
    """Data processing class."""

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig):
        self.df = pandas_df
        self.config = config

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

        x = self.df.drop(self.config.target, axis=1)
        y = self.df[self.config.target]
        return train_test_split(x, y, test_size=test_size, random_state=random_state, **kwargs)
