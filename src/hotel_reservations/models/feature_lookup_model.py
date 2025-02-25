import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession, DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_reservations.models._base import AbstractModel
from hotel_reservations.config import ProjectConfig, Tags


class FeatureLookUpModel(AbstractModel):
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        super().__init__(config, tags, spark)
        self.workspace = WorkspaceClient()
        self.fe = FeatureEngineeringClient()

        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.hotel_features"

        self.experiment_name = self.config.experiment_name_fe

    def create_feature_table(self):
        logger.info("Creating or updating feature table...")
        self.spark.sql(
            f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Booking_ID STRING NOT NULL, no_of_previous_cancellations INT, no_of_special_requests INT);
        """
        )

        logger.info("Adding primary key and enabling change data feed...")
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

        logger.info("Copy feature data from raw dataset into feature table")
        self.spark.sql(
            f"""
        INSERT INTO {self.feature_table_name} SELECT BOOKING_ID, no_of_previous_cancellations, 
        no_of_special_requests FROM {self.catalog_name}.{self.schema_name}.train_set
        """
        )

        self.spark.sql(
            f"""
        INSERT INTO {self.feature_table_name} SELECT BOOKING_ID, no_of_previous_cancellations, 
        no_of_special_requests FROM {self.catalog_name}.{self.schema_name}.test_set
        """
        )

        logger.info("Feature table creation completed successfully.")

    def load_data(self):
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            "no_of_previous_cancellations", "no_of_special_requests"
        )
        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.test_set"
        ).toPandas()

        # Need to cast type else it will throw error when coverting
        # spark dataframe into pandas dataframe
        self.train_set = self.train_set.withColumn(
            "Booking_ID", self.train_set["Booking_ID"].cast("string")
        )

    def feature_engineering(self):
        logger.info("Setting up feature engineering...")
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["no_of_previous_cancellations", "no_of_special_requests"],
                    lookup_key="Booking_ID",
                )
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        self.x_train = self.training_df.drop(self.target, axis=1)
        self.y_train = self.training_df[self.target]
        self.x_test = self.test_set.drop(self.target, axis=1)
        self.y_test = self.test_set[self.target]
        logger.info("Feature engineering completed successfully.")

    def prepare_model_pipeline(self):
        logger.info("Preparing model pipeline...")

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features),
            ],
            remainder="drop",
        )

        # Define the model
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LGBMClassifier(**self.model_parameters)),
            ]
        )
        logger.info("Model pipeline prepared successfully.")

        return pipeline

    def train(self):
        pipeline = self.prepare_model_pipeline()

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.x_train, self.y_train)
            y_pred = pipeline.predict(self.x_test)

            self._log_metrics(self.evaluate_model(y_pred, self.y_test))

            mlflow.log_param("model_type", "LightGBM with FE")
            mlflow.log_params(self.model_parameters)

            signature = infer_signature(self.x_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self):
        model_name = f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe"
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=model_name,
            tags=self.tags,
        )

        # Fetch latest registered version
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name, alias="latest-model", version=latest_version
        )

    def load_latest_model_and_predict(self, df: DataFrame, result_type: str):
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=df, result_type=result_type)
        return predictions
