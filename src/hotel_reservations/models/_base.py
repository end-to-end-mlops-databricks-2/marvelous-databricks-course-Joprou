import mlflow

from loguru import logger
from pyspark.sql import SparkSession
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from hotel_reservations.config import ProjectConfig, Tags


class AbstractModel:
    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession):
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name
        self.cat_features = config.cat_features
        self.num_features = config.num_features

        self.target = config.target
        self.config = config

        self.spark = spark
        self.tags = tags.model_dump()
        self.model_parameters = config.parameters

        self.run_id = None

    def evaluate_model(self, y_pred, y_test):
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label="Canceled")
        recall = recall_score(y_test, y_pred, pos_label="Canceled")
        precision = precision_score(y_test, y_pred, pos_label="Canceled")

        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"F1: {f1}")
        logger.info(f"Recall: {recall}")
        logger.info(f"Precision: {precision}")

        metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "recall": recall,
            "precision": precision,
        }

        return metrics

    def _log_metrics(self, metrics):
        logger.info("Logging metrics to MLflow...")
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        logger.info("Metrics logged successfully.")

    def _get_run_by_id(self, run_id: str = None):
        run_id = run_id or self.run_id

        if run_id is None:
            raise LookupError("Run ID must be defined to get run from MLFlow.")

        return mlflow.get_run(run_id)

    def retrieve_current_run_dataset(self):
        run = self._get_run_by_id(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        return dataset_source.load()

    def retrieve_current_run_metadata(self):
        run = self._get_run_by_id(self.run_id)
        data_dict = run.data.to_dictionary()
        metrics = data_dict["metrics"]
        params = data_dict["params"]
        return metrics, params
