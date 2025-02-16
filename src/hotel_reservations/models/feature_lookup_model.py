from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models._base import AbstractModel


class FeatureLookUpModel(AbstractModel):
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        super().__init__(config=config, tags=tags, spark=spark)

        self.workspace = WorkspaceClient()
        self.fe_client = FeatureEngineeringClient()

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.hotel_features"
        # self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_house_age"

        # MLflow
        self.experiment_name = self.config.experiment_name_fe

    def create_feature_table(self):
        self.spark.sql(
            f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Booking_ID STRING NOT NULL, no_of_previous_cancellations INT, no_of_special_requests INT);
        """
        )

        self.spark.sql(
            f"""
        ALTER TABLE {self.feature_table_name}
        ADD CONSTRAINT hotel_pk PRIMARY KEY(Booking_ID);
        """
        )
        self.spark.sql(
            f"""
        ALTER TABLE {self.feature_table_name}
        SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
        """
        )
