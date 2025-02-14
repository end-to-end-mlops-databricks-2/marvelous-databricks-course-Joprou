import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from loguru import logger
from pyspark.sql import SparkSession

from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models._base import AbstractModel


class BasicModel(AbstractModel):
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        super().__init__(config=config, tags=tags, spark=spark)

        self.experiment_name = config.experiment_name

    def load_data(self):
        """Load data from Databricks."""
        logger.info("Loading data from Databricks...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(
            f"{self.catalog_name}.{self.schema_name}.test_set"
        ).toPandas()
        self.data_version = "0"

        self.X_train = self.train_set[self.config.num_features + self.config.cat_features]
        self.y_train = self.train_set[self.config.target]
        self.X_test = self.test_set[self.config.num_features + self.config.cat_features]
        self.y_test = self.test_set[self.config.target]
        logger.info("Data loaded successfully.")

    def prepare_model_pipeline(self):
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

    def train_model(self):
        logger.info("Training model...")
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("Model trained successfully.")

    def log_model(self):
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

    def register_model(self):
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
        client.set_registered_model_alias(
            name=uc_model_name, alias="latest-model", version=latest_version
        )

        logger.info(f"`latest-model` tag is added to model version {latest_version}.")

    def load_latest_model_and_predict(
        self, input_data: pd.DataFrame, model_name: str = "hotel_reservations_model_basic"
    ):

        logger.info("Loading model...")
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.{model_name}@latest-model"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded succesfully.")

        return model.predict(input_data)
