import mlflow
from mlflow.models import infer_signature
from loguru import logger
from pyspark.sql import SparkSession

from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

from hotel_reservations.config import ProjectConfig, Tags


class BasicModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name
        self.cat_features = config.cat_features
        self.num_features = config.num_features
        self.target = config.target
        self.config = config
        self.tags = tags
        self.spark = spark
        self.model_parameters = config.parameters
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

    def evaluate_model(self):
        y_pred = self.pipeline.predict(self.X_test)

        # Evaluate
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, pos_label="Canceled")
        recall = recall_score(self.y_test, y_pred, pos_label="Canceled")
        precision = precision_score(self.y_test, y_pred, pos_label="Canceled")

        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"F1: {f1}")
        logger.info(f"Recall: {recall}")
        logger.info(f"Precision: {precision}")

        self.metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "recall": recall,
            "precision": precision,
        }

    def _log_metrics(self):
        logger.info("Logging metrics to MLflow...")
        for metric_name, metric_value in self.metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        logger.info("Metrics logged successfully.")

    def log_model(self):
        logger.info("Logging experiments to MLflow...")
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            y_pred = self.pipeline.predict(self.X_test)

            # Log parameters and metrics
            mlflow.log_param("model_type", "LGBMClassifier")
            mlflow.log_params(self.model_parameters)
            self._log_metrics()

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
            logger.info(f"Signature: {signature}")
            mlflow.sklearn.log_model(
                sk_model=self.pipeline, signature=signature, artifact_path="lightgbm-pipeline-model"
            )

            logger.info("Model logged successfully.")
