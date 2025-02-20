"""Utility functions for the hotel reservations application."""

import requests


def call_dbr_endpoint(
    dbr_host: str, dbr_token: str, endpoint_name: str, record: list[dict[str, any]]
) -> tuple[int, str]:
    """Call the Databricks serving endpoint.

    Args:
        dbr_host (str): Databricks host.
        dbr_token (str): Databricks token.
        endpoint_name (str): Serving endpoint name.
        record (list[dict[str, any]]): List of records to be sent to the serving endpoint.
    """
    serving_endpoint = f"https://{dbr_host}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {dbr_token}"},
        json={"dataframe_records": record},
        timeout=60,
    )

    return response.status_code, response.text
