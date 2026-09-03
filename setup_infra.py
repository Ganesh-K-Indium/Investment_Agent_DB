"""
Infrastructure Setup Script for Databricks 2-Agent SEC Intelligence.
Provisions:
1. Unity Catalog: Catalog, Schema, and Volume.
2. Delta Tables: sec_filing_chunks (with CDF) and agent_feedback (HITL Memory).
3. UC Governed Tools via DatabricksFunctionClient (tools/register_tools.py).
4. Databricks Vector Search endpoint (STANDARD).
"""

import sys
import os
import logging
from databricks.sdk import WorkspaceClient

# Ensure root workspace is in sys.path
_ROOT = os.path.abspath(os.path.dirname(__file__) if "__file__" in globals() else os.getcwd())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA,
    DATABRICKS_VOLUME,
    CHUNKS_TABLE,
    FEEDBACK_TABLE,
    VECTOR_SEARCH_ENDPOINT,
)
from tools.register_tools import register_all_uc_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("setup_infra")


def get_active_warehouse_id(w: WorkspaceClient) -> str:
    """Finds an active or usable SQL Warehouse ID."""
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No Databricks SQL Warehouse found in workspace. Please create or start a SQL Warehouse.")
    return warehouses[0].id


def initialize_governance_and_tables(w: WorkspaceClient, warehouse_id: str):
    """Provisions Catalog, Schema, Volume, and governed Delta tables."""
    logger.info("Setting up Catalog '%s', Schema '%s', Volume '%s'...", DATABRICKS_CATALOG, DATABRICKS_SCHEMA, DATABRICKS_VOLUME)

    statements = [
        f"CREATE CATALOG IF NOT EXISTS {DATABRICKS_CATALOG};",
        f"CREATE SCHEMA IF NOT EXISTS {DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA};",
        f"CREATE VOLUME IF NOT EXISTS {DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.{DATABRICKS_VOLUME};",
        f"""
        CREATE TABLE IF NOT EXISTS {FEEDBACK_TABLE} (
            feedback_id STRING NOT NULL,
            timestamp TIMESTAMP,
            ticker STRING,
            query STRING,
            rating STRING,
            feedback_text STRING,
            corrected_context STRING
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
            chunk_id STRING NOT NULL,
            ticker STRING,
            form_type STRING,
            year INT,
            quarter STRING,
            accession STRING,
            content STRING,
            CONSTRAINT sec_chunks_pk PRIMARY KEY (chunk_id)
        )
        TBLPROPERTIES (delta.enableChangeDataFeed = true);
        """,
        f"""
        ALTER TABLE {CHUNKS_TABLE} ADD COLUMNS IF NOT EXISTS (
            quarter STRING,
            accession STRING
        );
        """,
    ]

    for stmt in statements:
        lead_line = stmt.strip().splitlines()[0]
        logger.info("Executing: %s", lead_line)
        try:
            w.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                statement=stmt,
                wait_timeout="50s",
            )
        except Exception as exc:
            if "CREATE CATALOG" in lead_line:
                logger.info("Catalog '%s' already exists or managed externally (%s). Continuing...", DATABRICKS_CATALOG, exc)
            else:
                logger.error("Failed executing statement '%s': %s", lead_line, exc)
                raise
    logger.info("Unity Catalog objects and Delta tables initialized successfully.")


def ensure_vector_search_endpoint():
    """Validates or provisions the Databricks Vector Search endpoint."""
    from databricks.vector_search.client import VectorSearchClient

    logger.info("Validating Vector Search endpoint '%s'...", VECTOR_SEARCH_ENDPOINT)
    vsc = VectorSearchClient()

    endpoints = vsc.list_endpoints().get("endpoints", [])
    ep_names = [ep.get("name") for ep in endpoints]

    if VECTOR_SEARCH_ENDPOINT not in ep_names:
        logger.info("Endpoint '%s' not found. Creating STANDARD Vector Search endpoint...", VECTOR_SEARCH_ENDPOINT)
        vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT, endpoint_type="STANDARD")
        logger.info("Vector Search endpoint creation triggered.")
    else:
        logger.info("Vector Search endpoint '%s' exists and is ready.", VECTOR_SEARCH_ENDPOINT)


def main():
    logger.info("=== Initializing Production Databricks Infrastructure ===")
    w = WorkspaceClient()

    # 1. Locate Warehouse
    warehouse_id = get_active_warehouse_id(w)
    logger.info("Using SQL Warehouse: %s", warehouse_id)

    # 2. Provision Catalog, Schema, Volume, Delta Tables
    initialize_governance_and_tables(w, warehouse_id)

    # 3. Register UC Governed Tools
    logger.info("Registering UC functions with DatabricksFunctionClient...")
    register_all_uc_tools(catalog=DATABRICKS_CATALOG, schema=DATABRICKS_SCHEMA)

    # 4. Provision / Validate Vector Search Endpoint
    ensure_vector_search_endpoint()

    logger.info("=== Infrastructure setup completed successfully! ===")


if __name__ == "__main__":
    main()

