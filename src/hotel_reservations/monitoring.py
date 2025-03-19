"""Monitoring module."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)


def create_monitoring_table(
    spark: SparkSession,
    workspace: WorkspaceClient,
    catalog_schema: str,
    monitoring_table_name: str,
) -> None:
    """Create a monitoring table for model.

    Args:
        spark (SparkSession): Spark session.
        workspace (WorkspaceClient): Databricks workspace client.
        catalog_schema (str): Catalog schema name.
        monitoring_table_name (str): Monitoring table name.

    """

    logger.info(f"Creating monitoring table: {monitoring_table_name}...")

    workspace.quality_monitors.create(
        table_name=monitoring_table_name,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/victor/{monitoring_table_name}",
        output_schema_name=catalog_schema,
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
            prediction_col="prediction",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="booking_status",
        ),
    )

    spark.sql(f"ALTER TABLE {monitoring_table_name} " "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

    logger.info(f"Monitoring table created: {monitoring_table_name}.")


def create_or_refresh_monitoring_table(
    spark: SparkSession,
    workspace: WorkspaceClient,
    catalog_schema: str,
    inf_table_name: str,
    monitoring_table_name: str,
) -> None:
    """Create or refresh a monitoring table for model."

    Args:
        spark (SparkSession): Spark session.
        workspace (WorkspaceClient): Databricks workspace client.
        catalog_schema (str): Catalog schema name.
        inf_table_name (str): Inference table name.
        monitoring_table_name (str): Monitoring table name.

    """

    inf_table = spark.sql(f"SELECT * FROM {inf_table_name}")

    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("Booking_ID", StringType(), True),
                            StructField("no_of_adults", LongType(), True),
                            StructField("no_of_children", LongType(), True),
                            StructField("no_of_weekend_nights", LongType(), True),
                            StructField("no_of_week_nights", LongType(), True),
                            StructField("type_of_meal_plan", StringType(), True),
                            StructField("required_car_parking_space", LongType(), True),
                            StructField("room_type_reserved", StringType(), True),
                            StructField("lead_time", LongType(), True),
                            StructField("arrival_year", LongType(), True),
                            StructField("arrival_month", LongType(), True),
                            StructField("arrival_date", LongType(), True),
                            StructField("market_segment_type", StringType(), True),
                            StructField("repeated_guest", LongType(), True),
                            StructField("no_of_previous_cancellations", LongType(), True),
                            StructField("no_of_previous_bookings_not_canceled", LongType(), True),
                            StructField("avg_price_per_room", DoubleType(), True),
                            StructField("no_of_special_requests", LongType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    response_schema = StructType(
        [
            StructField("predictions", ArrayType(StringType()), True),
            StructField(
                "databricks_output",
                StructType(
                    [
                        StructField("trace", StringType(), True),
                        StructField("databricks_request_id", StringType(), True),
                    ]
                ),
                True,
            ),
        ]
    )

    inf_table_parsed = inf_table.withColumn(
        "parsed_request", func.from_json(func.col("request"), request_schema)
    ).withColumn(
        "parsed_response",
        func.from_json(func.col("response"), response_schema),
    )

    df_exploded = inf_table_parsed.withColumn("record", func.explode(func.col("parsed_request.dataframe_records")))

    # COMMAND ----------
    df_final = df_exploded.select(
        func.from_unixtime(func.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
        "timestamp_ms",
        "databricks_request_id",
        "execution_time_ms",
        func.col("record.Booking_ID").alias("Booking_ID"),
        func.col("record.no_of_adults").alias("no_of_adults"),
        func.col("record.no_of_children").alias("no_of_children"),
        func.col("record.no_of_weekend_nights").alias("no_of_weekend_nights"),
        func.col("record.no_of_week_nights").alias("no_of_week_nights"),
        func.col("record.type_of_meal_plan").alias("type_of_meal_plan"),
        func.col("record.required_car_parking_space").alias("required_car_parking_space"),
        func.col("record.room_type_reserved").alias("room_type_reserved"),
        func.col("record.lead_time").alias("lead_time"),
        func.col("record.arrival_year").alias("arrival_year"),
        func.col("record.arrival_month").alias("arrival_month"),
        func.col("record.arrival_date").alias("arrival_date"),
        func.col("record.market_segment_type").alias("market_segment_type"),
        func.col("record.repeated_guest").alias("repeated_guest"),
        func.col("record.no_of_previous_cancellations").alias("no_of_previous_cancellations"),
        func.col("record.no_of_previous_bookings_not_canceled").alias("no_of_previous_bookings_not_canceled"),
        func.col("record.avg_price_per_room").alias("avg_price_per_room"),
        func.col("record.no_of_special_requests").alias("no_of_special_requests"),
        func.col("parsed_response.predictions")[0].alias("prediction"),
        func.lit("hotel-reservations-model-basic").alias("model_name"),
    )

    test_set = spark.table(f"{catalog_schema}.test_set")
    inference_set_skewed = spark.table(f"{catalog_schema}.test_set")

    df_final_with_status = (
        df_final.join(test_set.select("Booking_ID", "booking_status"), on="Booking_ID", how="left")
        .withColumnRenamed("booking_status", "booking_status_test")
        .join(inference_set_skewed.select("Booking_ID", "booking_status"), on="Booking_ID", how="left")
        .withColumnRenamed("booking_status", "booking_status_inference")
    )

    df_final_with_status = (
        df_final_with_status.select(
            "*",
            func.coalesce(func.col("booking_status_test"), func.col("booking_status_inference")).alias(
                "booking_status"
            ),
        )
        .drop("booking_status_test", "booking_status_inference")
        .dropna(subset=["booking_status", "prediction"])
    )

    logger.info(f"Writing to monitoring table: {monitoring_table_name}...")
    df_final_with_status.write.mode("append").saveAsTable(monitoring_table_name)

    try:
        workspace.quality_monitors.get(monitoring_table_name)
        logger.info(f"{monitoring_table_name} found. Refreshing table...")
        workspace.quality_monitors.run_refresh(monitoring_table_name)
        logger.info(f"{monitoring_table_name} refreshed.")
    except NotFound:
        logger.info(f"{monitoring_table_name} not found.")
        create_monitoring_table(
            spark,
            workspace,
            catalog_schema,
            monitoring_table_name,
        )
