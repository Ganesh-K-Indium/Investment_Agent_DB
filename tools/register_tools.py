"""
Registration script for Unity Catalog governed tools.
Uses DatabricksFunctionClient from unitycatalog-ai to publish Python tools
as governed UC functions under {CATALOG}.{SCHEMA} with replace=True.
"""

import sys
import os
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DATABRICKS_CATALOG, DATABRICKS_SCHEMA
from tools.uc_tools import check_filing_status, record_feedback, get_relevant_feedback

logger = logging.getLogger("tools.register_tools")


def register_all_uc_tools(catalog: str = DATABRICKS_CATALOG, schema: str = DATABRICKS_SCHEMA):
    """
    Registers the SEC Intelligence tools into Unity Catalog via DatabricksFunctionClient.
    Includes active notebook Spark session sync and SQL Warehouse fallback for guaranteed registration.
    """
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient, generate_sql_function_body
    from databricks.sdk import WorkspaceClient

    logger.info("Initializing DatabricksFunctionClient for UC tool registration...")
    client = DatabricksFunctionClient()

    # If running inside a Databricks Notebook with an active SparkSession, bind it directly to prevent detached Connect sessions
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is not None:
            logger.info("Synchronizing active notebook Spark session for %s.%s...", catalog, schema)
            spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
            spark.sql(f"USE CATALOG {catalog}")
            spark.sql(f"USE SCHEMA {schema}")
            client.set_spark_session(spark)
    except Exception as exc:
        logger.debug("Active notebook Spark sync: %s", exc)

    tools_to_register = [
        check_filing_status,
        record_feedback,
        get_relevant_feedback,
    ]

    registered_functions = []

    for tool_func in tools_to_register:
        func_name = tool_func.__name__
        full_name = f"{catalog}.{schema}.{func_name}"
        logger.info("Registering function '%s' to UC %s (replace=True)...", func_name, full_name)
        try:
            client.create_python_function(
                func=tool_func,
                catalog=catalog,
                schema=schema,
                replace=True,
            )
            logger.info("Successfully registered UC function: %s", full_name)
            registered_functions.append(full_name)
        except Exception as exc:
            logger.warning(
                "DatabricksFunctionClient direct creation encountered: %s. Attempting SQL Warehouse fallback...",
                exc,
            )
            try:
                sql_body = generate_sql_function_body(tool_func, catalog, schema, replace=True)
                w = WorkspaceClient()
                warehouses = list(w.warehouses.list())
                if not warehouses:
                    raise RuntimeError("No SQL Warehouse available for fallback function creation.")
                w.statement_execution.execute_statement(
                    warehouse_id=warehouses[0].id,
                    statement=sql_body,
                    wait_timeout="50s",
                )
                logger.info("Successfully registered UC function via SQL Warehouse: %s", full_name)
                registered_functions.append(full_name)
            except Exception as fb_exc:
                logger.error("Failed to register UC function '%s': %s (fallback error: %s)", func_name, exc, fb_exc)
                raise

    return registered_functions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        registered = register_all_uc_tools()
        print(f"\n[UC TOOLS REGISTERED] Successfully registered {len(registered)} functions:")
        for r in registered:
            print(f" - {r}")
    except Exception as e:
        print(f"\n[UC TOOLS REGISTRATION FAILED] {e}", file=sys.stderr)
        sys.exit(1)

