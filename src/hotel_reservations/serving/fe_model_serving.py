"""Feature lookup serving module."""

from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy

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
