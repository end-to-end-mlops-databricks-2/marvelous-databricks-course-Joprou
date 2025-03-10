"""Feature lookup serving module."""

import time

from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
from databricks.sdk.service.pipelines import UpdateInfoState
from loguru import logger

from hotel_reservations.serving.model_serving import ModelServing


class FeatureLookupServing(ModelServing):
    """Feature lookup serving class."""

    def __init__(
        self,
        model_name: str,
        endpoint_name: str,
        feature_table_name: str,
        feature_table_name_online: str = None,
    ):
        super().__init__(model_name, endpoint_name)

        self.feature_table_name = feature_table_name
        self.feature_table_name_online = feature_table_name_online or f"{feature_table_name}_online"

    def create_online_table(self):
        """Create online table based on feature table."""
        spec = OnlineTableSpec(
            primary_key_columns=["Booking_ID"],
            source_table_full_name=self.feature_table_name,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
            perform_full_copy=False,
        )

        self.workspace.online_tables.create(name=self.feature_table_name_online, spec=spec)

    def update_online_table(self, pipeline_id: str) -> None:
        """Update online table with the pipeline."""
        update_response = self.workspace.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)

        while True:
            update_info = self.workspace.pipelines.get_update(
                pipeline_id=pipeline_id, update_id=update_response.update_id
            )
            state = update_info.update.state

            if state == UpdateInfoState.COMPLETED:
                logger.info(f"Online table updated successfully with pipeline {pipeline_id}.")
                break
            if state in [UpdateInfoState.FAILED, UpdateInfoState.CANCELED]:
                msg = f"Online table update failed with pipeline {pipeline_id}."
                logger.error(msg)
                raise SystemError(msg)
            if state == UpdateInfoState.WAITING_FOR_RESOURCES:
                logger.warning("Waiting for resources to update online table.")
            else:
                logger.info(f"Online table update state: {state.value}")

            time.sleep(30)
