# Databricks notebook source
# %pip install --force-reinstall hotel_reservations-0.0.1-py3-none-any.whl
# dbutils.library.restartPython()


# COMMAND ----------
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
from pyspark.sql import SparkSession

from hotel_reservations.config import ProjectConfig

spark = SparkSession.builder.config("spark.sql.session.timeZone", "UTC").getOrCreate()

workspace = WorkspaceClient()
sources = workspace.data_sources.list()

config = ProjectConfig.from_yaml("../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
CATALOG_SCHEMA = f"{catalog_name}.{schema_name}"
ALERT_OPERAND_COL = "low_performance_percentage"
ALERT_METRIC = "f1_score.weighted"

# COMMAND ----------
ALERT_QUERY = f"""
SELECT
  (COUNT(CASE WHEN {ALERT_METRIC} < 0.8 THEN 1 END) * 100.0 /
   COUNT(CASE WHEN {ALERT_METRIC} IS NOT NULL AND NOT isnan({ALERT_METRIC}) THEN 1 END))
   AS {ALERT_OPERAND_COL}
FROM {CATALOG_SCHEMA}.hotel_reservations_monitoring_profile_metrics;

"""

query = workspace.queries.create(
    query=sql.CreateQueryRequestQuery(
        display_name=f"hotel-reservations-alert-query-{time.time_ns()}",
        warehouse_id=sources[0].warehouse_id,
        description="Alert on house price model F1 score",
        query_text=ALERT_QUERY,
    )
)

alert = workspace.alerts.create(
    alert=sql.CreateAlertRequestAlert(
        condition=sql.AlertCondition(
            operand=sql.AlertConditionOperand(
                column=sql.AlertOperandColumn(name=ALERT_OPERAND_COL),
            ),
            op=sql.AlertOperator.GREATER_THAN,
            threshold=sql.AlertConditionThreshold(value=sql.AlertOperandValue(double_value=45)),
        ),
        display_name=f"hotel-reservations-f1score-alert-{time.time_ns()}",
        query_id=query.id,
    )
)


# COMMAND ----------
# cleanup
workspace.queries.delete(id=query.id)
workspace.alerts.delete(id=alert.id)

# COMMAND ----------
