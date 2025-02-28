"""This module contains the FeatureServing class."""

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from loguru import logger


class FeatureServing:
    """Feature serving class."""

    def __init__(self, feature_table_name: str, feature_spec_name: str, endpoint_name: str):
        self.workspace = WorkspaceClient()
        self.feature_table_name = feature_table_name
        self.feature_spec_name = feature_spec_name
        self.endpoint_name = endpoint_name
        self.online_table_name = f"{feature_table_name}_online"
        self.fe = FeatureEngineeringClient()

    def create_online_table(self):
        """Create online table based on feature table."""

        logger.info(
            f"Creating online table {self.online_table_name}" f"using {self.feature_table_name} as source table"
        )

        spec = OnlineTableSpec(
            primary_key_columns=["Booking_ID"],
            source_table_full_name=self.feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
            perform_full_copy=False,
        )

        self.workspace.online_tables.create(name=self.online_table_name, spec=spec)

        logger.info(f"Online table {self.online_table_name} created successfully")

    def create_feature_spec(self) -> None:
        """Create feature spec for serving."""
        features = [
            FeatureLookup(
                table_name=self.feature_table_name,
                lookup_key="Booking_ID",
                # feature_names=["no_of_previous_cancellations", "no_of_special_requests"],
            )
        ]

        self.fe.create_feature_spec(name=self.feature_spec_name, features=features)

    def deploy_or_update_serving_endpoint(self, workload_size: str = "Small", scale_to_zero: bool = True) -> None:
        """Deploy or update the serving endpoint.

        Args:
            workload_size (str, optional): Workload size of the serving endpoint. \
                Defaults to "Small".
            scale_to_zero (bool, optional): Enable scale to zero. Defaults to True.
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())

        served_entities = [
            ServedEntityInput(
                entity_name=self.feature_spec_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
            )
        ]

        if not endpoint_exists:
            self.workspace.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(served_entities=served_entities),
            )
        else:
            self.workspace.serving_endpoints.update_config(name=self.endpoint_name, served_entities=served_entities)
