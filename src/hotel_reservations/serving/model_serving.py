"""Model serving class to deploy or update the serving endpoint."""

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from loguru import logger


class ModelServing:
    """Model serving class."""

    def __init__(self, model_name: str, endpoint_name: str):
        self.workspace = WorkspaceClient()
        self.endpoint_name = endpoint_name
        self.model_name = model_name

    def get_latest_model_version(self) -> str:
        """Fetch the latest version of the model from MLflow registry.

        Returns:
            str: Latest version of the model.
        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        logger.info(f"Latest version of the `{self.model_name} model: {latest_version}")
        return latest_version

    def deploy_or_update_serving_endpoint(
        self, version: str = "latest", workload_size: str = "Small", scale_to_zero: bool = True
    ) -> None:
        """Deploy or update the serving endpoint.

        Args:
            version (str, optional): Model version to be deployed. Defaults to "latest".
            workload_size (str, optional): Workload size of the serving endpoint. \
                Defaults to "Small".
            scale_to_zero (bool, optional): Enable scale to zero. Defaults to True.
        """

        if version == "latest":
            logger.info("Deploying the latest version of the model")
            entity_version = self.get_latest_model_version()
        else:
            logger.info(f"Deploying the model version: {version}")
            entity_version = version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())

        if not endpoint_exists:
            logger.info(f"Creating a new serving endpoint: {self.endpoint_name}")
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(served_entities=served_entities),
            )
        else:
            logger.info(f"Updating the existing serving endpoint: {self.endpoint_name}")
            self.workspace.serving_endpoints.update_config(
                name=self.endpoint_name,
                served_entities=served_entities,
            )
        logger.info(f"Endpoint `{self.endpoint_name}` deployed successfully.")
