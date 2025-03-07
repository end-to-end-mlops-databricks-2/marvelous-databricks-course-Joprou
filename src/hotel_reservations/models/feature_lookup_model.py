import mlflow
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models._base import AbstractModel


class FeatureLookUpModel(AbstractModel):
    """Model that uses feature lookup."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        super().__init__(config, tags, spark)
        self.workspace = WorkspaceClient()
        self.fe = FeatureEngineeringClient()

        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.hotel_features"

        self.experiment_name = self.config.experiment_name_fe

    def create_feature_table(self) -> None:
        """Create feature table from raw dataset."""

        logger.info("Creating or updating feature table...")
        self.spark.sql(
            f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Booking_ID STRING NOT NULL, no_of_previous_cancellations LONG,
        no_of_special_requests LONG);
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
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        # Need to cast type else it will throw error when coverting
        # spark dataframe into pandas dataframe
        self.train_set = self.train_set.withColumn("Booking_ID", self.train_set["Booking_ID"].cast("string"))

    def feature_engineering(self):
        """Setup feature engineering dataset."""

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
        """Prepare the model pipeline."""
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
        """Train and log the model to MLflow."""
        pipeline = self.prepare_model_pipeline()

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.x_train, self.y_train)
            logger.info("Model trained successfully.")
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

            logger.info("Model logged successfully.")

    def register_model(self) -> int:
        """Register the trained model and set an alias to the latest version.'

        Returns:
            int: Latest version of the registered model.

        """
        model_name = f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_fe"
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=model_name,
            tags=self.tags,
        )

        # Fetch latest registered version
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(name=model_name, alias="latest-model", version=latest_version)

        return latest_version

    def load_latest_model_and_predict(self, df: DataFrame, result_type: str) -> DataFrame:
        """Load the latest registered model and make predictions on the input data.

        Args:
            df (DataFrame): Input data to make predictions on.
            result_type (str): Type of result to return. \
                Args for score_batch method from FeatureEngineeringClient.

        Returns:
            DataFrame: Predictions made by the model.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}." "hotel_reservations_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=df, result_type=result_type)
        return predictions

    def update_feature_table(self):
        """Update feature table with latest data from raw dataset."""
        logger.info("Updating feature table...")

        query_base = """
        WITH max_timestamp AS (
            SELECT MAX(update_timestamp_utc) AS max_timestamp
            FROM {catalog_name}.{schema_name}.{dataset_type}
        )
        INSERT INTO {feature_table_name}
        SELECT BOOKING_ID, no_of_previous_cancellations, no_of_special_requests
        FROM {catalog_name}.{schema_name}.{dataset_type}
        WHERE update_timestamp_utc = (SELECT max_timestamp FROM max_timestamp)
        """

        for dataset_type in ["train_set", "test_set"]:
            self.spark.sql(
                query_base.format(
                    catalog_name=self.catalog_name,
                    schema_name=self.schema_name,
                    dataset_type=dataset_type,
                    feature_table_name=self.feature_table_name,
                )
            )
        logger.info("Feature table updated successfully.")

    def create_or_update_feature_table(self):
        """Create or update feature table based on existence."""
        if self.spark.catalog.tableExists(self.feature_table_name):
            logger.info(f"Feature table {self.feature_table_name} already exists.")
            self.update_feature_table()

        else:
            logger.info(f"Feature table {self.feature_table_name} does not exist.")
            self.create_feature_table()

    def is_model_improves(self, test_set: DataFrame) -> bool:
        """Check if the current model is better than the latest registered model.

        Args:
            test_set (DataFrame): Test set to evaluate the model.

        Returns:
            bool: True if the current model is better than the latest registered model.
        """

        x_test = test_set.drop(self.config.target)

        # convert y_true to pandas Series for evaluation
        y_test = test_set.select(self.config.target).toPandas()[self.config.target]

        # predict using latest registered model
        prediction_latest = self.load_latest_model_and_predict(x_test, result_type="string").toPandas()["prediction"]
        metrics_latest = self.evaluate_model(prediction_latest, y_test)
        f1_latest = metrics_latest["f1"]

        # predict using current trained model
        current_model_uri = f"runs:/{self.run_id}/lightgbm-pipeline-model-fe"
        prediction_current = self.fe.score_batch(
            model_uri=current_model_uri, df=x_test, result_type="string"
        ).toPandas()["prediction"]
        metrics_current = self.evaluate_model(prediction_current, y_test)
        f1_current = metrics_current["f1"]

        logger.info(f"F1 score of latest model: {f1_latest}")
        logger.info(f"F1 score of current model: {f1_current}")

        if f1_current < f1_latest:
            logger.info("Current model is worse than the latest registered model.")
            return False
        if f1_current == f1_latest:
            logger.info("Current model is as good as the latest registered model.")
            return False

        logger.info("Current model is better than the latest registered model.")
        return True
