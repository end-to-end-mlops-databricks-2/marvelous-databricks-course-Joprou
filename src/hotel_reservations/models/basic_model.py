"""Basic model class for hotel reservations project."""

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models._base import AbstractModel


class BasicModel(AbstractModel):
    """Basic model class."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        super().__init__(config=config, tags=tags, spark=spark)

        self.experiment_name = config.experiment_name

    def load_data(self):
        """Load data from Databricks."""
        logger.info("Loading data from Databricks...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.data_version = "0"

        self.X_train = self.train_set[self.config.num_features + self.config.cat_features]
        self.y_train = self.train_set[self.config.target]
        self.X_test = self.test_set[self.config.num_features + self.config.cat_features]
        self.y_test = self.test_set[self.config.target]
        logger.info("Data loaded successfully.")

    def prepare_model_pipeline(self) -> None:
        """Setup sklearn model pipeline"""
        logger.info("Preparing model pipeline...")

        # Bundle preprocessing for numerical and categorical data
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features),
            ],
            remainder="drop",
        )

        # Define the model
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("classifier", LGBMClassifier(**self.model_parameters)),
            ]
        )
        logger.info("Model pipeline prepared successfully.")

    def train_model(self) -> None:
        """Train the model."""
        logger.info("Training model...")
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("Model trained successfully.")

    def log_model(self):
        """Log experiment to MLflow."""
        logger.info("Logging experiments to MLflow...")
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            y_pred = self.pipeline.predict(self.X_test)

            # Log parameters and metrics
            mlflow.log_param("model_type", "LGBMClassifier")
            mlflow.log_params(self.model_parameters)
            self._log_metrics(self.evaluate_model(y_pred, self.y_test))

            # Log dataset
            logger.info("Logging dataset to MLflow...")
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")
            logger.info("Dataset logged successfully.")

            # Log model
            logger.info("Logging model to MLflow...")
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)
            mlflow.sklearn.log_model(
                sk_model=self.pipeline, signature=signature, artifact_path="lightgbm-pipeline-model"
            )

            logger.info("Model logged successfully.")

    def register_model(self) -> int:
        """Register model in Unity Catalog.

        Returns:
            int: Version of the registered model.
        """
        logger.info("Registering model in Unity Catalog....")

        uc_model_name = f"{self.catalog_name}.{self.schema_name}.hotel_reservations_model_basic"

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
            name=uc_model_name,
            tags=self.tags,
        )

        latest_version = registered_model.version

        logger.info(f"Model registered as version {latest_version}.")

        client = MlflowClient()
        client.set_registered_model_alias(name=uc_model_name, alias="latest-model", version=latest_version)

        logger.info(f"`latest-model` tag is added to model version {latest_version}.")

        return latest_version

    def load_latest_model_and_predict(
        self, input_data: pd.DataFrame, model_name: str = "hotel_reservations_model_basic"
    ) -> np.array:
        """Load latest model from Unity Catalog and predict on input data.

        Args:
            input_data (pd.DataFrame): Input data to predict.
            model_name (str, optional): Model name to load.\
                Defaults to "hotel_reservations_model_basic".

        Returns:
            np.array: Predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.{model_name}@latest-model"
        logger.info(f"Loading model from URI: {model_uri}...")
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded succesfully.")

        return model.predict(input_data)

    def is_model_improves(self, test_set: DataFrame) -> bool:
        """Check if the current model is better than the latest registered model.

        Args:
            test_set (DataFrame): Test set to evaluate the model.

        Returns:
            bool: True if the current model is better than the latest registered model.
        """
        test_set = test_set.toPandas()

        x_test = test_set.drop(self.config.target, axis=1)
        y_test = test_set[self.config.target]

        # predict using latest registered model
        logger.info("=" * 20)
        logger.info("Predicting using the latest registered model...")
        prediction_latest = self.load_latest_model_and_predict(x_test)
        metrics_latest = self.evaluate_model(prediction_latest, y_test)
        f1_latest = metrics_latest["f1"]
        logger.info("=" * 20)

        # predict using current trained
        logger.info("=" * 20)
        logger.info("Predicting using the current model...")
        current_model_uri = f"runs:/{self.run_id}/lightgbm-pipeline-model"
        logger.info(f"Loading model from URI: {current_model_uri}...")
        model = mlflow.sklearn.load_model(current_model_uri)
        logger.info("Model loaded succesfully.")
        prediction_current = model.predict(x_test)
        metrics_current = self.evaluate_model(prediction_current, y_test)
        f1_current = metrics_current["f1"]
        logger.info("=" * 20)

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
