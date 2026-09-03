"""
Databricks Vector Indexer with Deduplication Prevention and Idempotent Ingestion.
Extracts metadata from filing text headers and filenames, chunks content deterministically,
upserts chunks into Delta Table with Change Data Feed (CDF) enabled (preventing duplicates),
and triggers Databricks Vector Search Delta Sync.
"""

import os
import re
import logging
from typing import List, Dict, Optional, Tuple

from config import (
    CHUNKS_TABLE,
    VS_INDEX_NAME,
    VECTOR_SEARCH_ENDPOINT,
    EMBEDDING_MODEL_ENDPOINT,
)

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def extract_filing_metadata_from_text(content: str, fallback_file_path: str) -> Dict[str, str]:
    """
    Extracts structured metadata (ticker, form_type, year, quarter, accession)
    from standardized filing header lines or filename.
    """
    meta = {
        "ticker": "UNKNOWN",
        "form_type": "UNKNOWN",
        "year": "0",
        "quarter": "N/A",
        "accession": "N/A",
    }

    # 1. Inspect header
    for line in content[:2000].splitlines():
        if line.startswith("TICKER:"):
            meta["ticker"] = line.split(":", 1)[1].strip().upper()
        elif line.startswith("FORM_TYPE:"):
            meta["form_type"] = line.split(":", 1)[1].strip().upper()
        elif line.startswith("YEAR:"):
            meta["year"] = line.split(":", 1)[1].strip()
        elif line.startswith("QUARTER:"):
            meta["quarter"] = line.split(":", 1)[1].strip()
        elif line.startswith("ACCESSION_NUMBER:"):
            meta["accession"] = line.split(":", 1)[1].strip()

    # 2. Fallback to filename if header missing
    if meta["ticker"] == "UNKNOWN" or meta["accession"] == "N/A":
        basename = os.path.basename(fallback_file_path)
        stem, _ = os.path.splitext(basename)
        parts = stem.split("_")
        if len(parts) >= 4:
            meta["ticker"] = parts[0].upper()
            meta["form_type"] = parts[1].upper()
            meta["year"] = parts[2]
            meta["accession"] = parts[3]
        elif len(parts) >= 3:
            meta["ticker"] = parts[0].upper()
            meta["form_type"] = parts[1].upper()
            meta["year"] = parts[2]

    return meta


def chunk_text_deterministically(
    text: str,
    ticker: str,
    form_type: str,
    year: int,
    quarter: str = "N/A",
    accession: str = "",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Splits text into overlapping segments with deterministic chunk IDs.
    Deterministic format: {ticker}_{form_type}_{accession}_{chunk_index}
    Guarantees idempotency and prevents duplicate chunks upon repeated ingestion.
    """
    chunks = []
    # Strip metadata header if present to avoid indexing header boilerplate repeatedly
    body_text = text
    if "================================================================================" in text:
        body_text = text.split("================================================================================", 1)[-1].strip()

    text_len = len(body_text)
    start = 0
    step = chunk_size - chunk_overlap
    chunk_idx = 0

    clean_accession = accession.replace("-", "").strip() or "0000000000"
    clean_quarter = (quarter or "N/A").strip()

    while start < text_len:
        end = min(start + chunk_size, text_len)
        content_segment = body_text[start:end].strip()
        if content_segment:
            chunk_id = f"{ticker.upper()}_{form_type.upper()}_{clean_accession}_{chunk_idx:05d}"
            chunks.append({
                "chunk_id": chunk_id,
                "ticker": ticker.upper(),
                "form_type": form_type.upper(),
                "year": int(year),
                "quarter": clean_quarter,
                "accession": clean_accession,
                "content": content_segment,
            })
            chunk_idx += 1
        start += step

    return chunks


def load_and_chunk_file(file_path: str) -> Tuple[Dict[str, str], List[Dict]]:
    """Reads a filing file, extracts metadata, and chunks content deterministically."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    meta = extract_filing_metadata_from_text(content, file_path)
    chunks = chunk_text_deterministically(
        text=content,
        ticker=meta["ticker"],
        form_type=meta["form_type"],
        year=int(meta["year"]) if meta["year"].isdigit() else 0,
        quarter=meta.get("quarter", "N/A"),
        accession=meta["accession"],
    )
    return meta, chunks


def write_chunks_to_delta_table(chunks: List[Dict]) -> int:
    """
    Upserts chunks into Delta table {CATALOG}.{SCHEMA}.sec_filing_chunks with CDF enabled.
    Uses MERGE on chunk_id to strictly prevent duplicate entries.
    """
    if not chunks:
        logger.info("No chunks to write to Delta table.")
        return 0

    # 1. Check for PySpark session (Native Databricks Cluster / Notebook)
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is None:
            try:
                spark = SparkSession.builder.getOrCreate()
            except Exception:
                spark = None

        if spark is not None:
            logger.info("Writing %d chunks via SparkSession to %s (idempotent merge)...", len(chunks), CHUNKS_TABLE)
            new_df = spark.createDataFrame(chunks)
            new_df.createOrReplaceTempView("new_incoming_chunks")

            spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
                chunk_id STRING NOT NULL,
                ticker STRING,
                form_type STRING,
                year INT,
                quarter STRING,
                accession STRING,
                content STRING,
                CONSTRAINT sec_chunks_pk PRIMARY KEY (chunk_id)
            ) TBLPROPERTIES (delta.enableChangeDataFeed = true);
            """)

            spark.sql(f"""
            MERGE INTO {CHUNKS_TABLE} AS target
            USING new_incoming_chunks AS source
            ON target.chunk_id = source.chunk_id
            WHEN MATCHED THEN UPDATE SET target.content = source.content, target.quarter = source.quarter
            WHEN NOT MATCHED THEN INSERT *;
            """)
            logger.info("Successfully merged %d chunks via Spark into %s.", len(chunks), CHUNKS_TABLE)
            return len(chunks)
    except Exception as e:
        logger.info("Spark cluster session not active (%s). Using Databricks SQL Warehouse.", e)

    # 2. Databricks SQL Warehouse Execution
    from config import get_workspace_client

    w = get_workspace_client()
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        try:
            warehouses = list(w.warehouses.list())
            if warehouses:
                warehouse_id = warehouses[0].id
        except Exception as exc:
            logger.warning("Could not list warehouses (%s)", exc)

    if not warehouse_id:
        raise RuntimeError("No Databricks SQL Warehouse available. Please configure DATABRICKS_WAREHOUSE_ID in app.yaml.")

    logger.info("Using SQL Warehouse ID: %s for Delta table ingestion", warehouse_id)

    # Ensure table exists with CDF enabled
    create_sql = f"""
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
    """
    w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=create_sql,
        wait_timeout="50s",
    )

    # Schema evolution: ensure quarter column exists on pre-existing tables
    try:
        w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=f"ALTER TABLE {CHUNKS_TABLE} ADD COLUMNS (quarter STRING);",
            wait_timeout="20s",
        )
    except Exception:
        pass  # Column already exists

    # Upsert in batches of 40 using MERGE INTO
    batch_size = 40
    total_processed = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        select_rows = []
        for c in batch:
            cid = c["chunk_id"].replace("'", "''")
            tkr = c["ticker"].replace("'", "''")
            frm = c["form_type"].replace("'", "''")
            yr = int(c["year"])
            qtr = c.get("quarter", "N/A").replace("'", "''")
            acc = c["accession"].replace("'", "''")
            cnt = c["content"].replace("'", "''").replace("\\", "\\\\")
            select_rows.append(f"SELECT '{cid}' AS chunk_id, '{tkr}' AS ticker, '{frm}' AS form_type, {yr} AS year, '{qtr}' AS quarter, '{acc}' AS accession, '{cnt}' AS content")

        source_union = " UNION ALL ".join(select_rows)
        merge_sql = f"""
        MERGE INTO {CHUNKS_TABLE} AS target
        USING ({source_union}) AS source
        ON target.chunk_id = source.chunk_id
        WHEN MATCHED THEN UPDATE SET target.content = source.content, target.quarter = source.quarter
        WHEN NOT MATCHED THEN INSERT (chunk_id, ticker, form_type, year, quarter, accession, content)
        VALUES (source.chunk_id, source.ticker, source.form_type, source.year, source.quarter, source.accession, source.content);
        """
        resp = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=merge_sql,
            wait_timeout="50s",
        )
        if hasattr(resp, "status") and resp.status:
            state_str = str(getattr(resp.status, "state", ""))
            if "FAILED" in state_str:
                err_text = getattr(resp.status.error, "message", "Unknown SQL execution failure")
                raise RuntimeError(f"Delta table MERGE failed: {err_text}")
        total_processed += len(batch)

    logger.info("Successfully merged %d chunks into %s without duplicates.", total_processed, CHUNKS_TABLE)
    return total_processed


def get_vector_search_client():
    """
    Initializes VectorSearchClient using the official Databricks authentication precedence:
    1. Personal Access Token (PAT) if DATABRICKS_TOKEN provided.
    2. Azure / AWS Service Principal credentials (DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET, DATABRICKS_HOST).
    3. Auto-detection (Databricks notebook context).
    """
    from databricks.vector_search.client import VectorSearchClient

    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    client_id = os.getenv("DATABRICKS_CLIENT_ID")
    client_secret = os.getenv("DATABRICKS_CLIENT_SECRET")
    tenant_id = (
        os.getenv("DATABRICKS_AZURE_TENANT_ID")
        or os.getenv("AZURE_TENANT_ID")
        or os.getenv("ARM_TENANT_ID")
    )

    if token and host:
        return VectorSearchClient(workspace_url=host, personal_access_token=token)
    elif client_id and client_secret and host:
        kwargs = {
            "workspace_url": host,
            "service_principal_client_id": client_id,
            "service_principal_client_secret": client_secret,
        }
        if tenant_id:
            kwargs["azure_tenant_id"] = tenant_id
        return VectorSearchClient(**kwargs)
    else:
        return VectorSearchClient()


def sync_vector_search_index() -> str:
    """Synchronizes or provisions the Databricks Vector Search Delta Sync index."""
    # 1. Prefer native Databricks SDK (pre-installed on all Databricks runtimes, zero pip dependencies)
    try:
        from config import get_workspace_client
        w = get_workspace_client()
        logger.info("Triggering vector search sync via native Databricks SDK on '%s'...", VS_INDEX_NAME)
        w.vector_search_indexes.sync_index(index_name=VS_INDEX_NAME)
        logger.info("Successfully triggered vector index sync on '%s'.", VS_INDEX_NAME)
        return VS_INDEX_NAME
    except Exception as sdk_exc:
        logger.info("SDK vector sync deferred (%s). Trying VectorSearchClient...", sdk_exc)

    # 2. Fallback to databricks.vector_search client if installed
    try:
        vsc = get_vector_search_client()

        endpoints = vsc.list_endpoints().get("endpoints", [])
        ep_names = [ep.get("name") for ep in endpoints]

        if VECTOR_SEARCH_ENDPOINT not in ep_names:
            logger.info("Creating Vector Search endpoint '%s'...", VECTOR_SEARCH_ENDPOINT)
            vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT, endpoint_type="STANDARD")
        else:
            logger.info("Vector Search endpoint '%s' verified.", VECTOR_SEARCH_ENDPOINT)

        try:
            index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=VS_INDEX_NAME)
            logger.info("Triggering sync on index '%s'...", VS_INDEX_NAME)
            index.sync()
            return VS_INDEX_NAME
        except Exception:
            logger.info("Creating new DELTA_SYNC index %s...", VS_INDEX_NAME)
            vsc.create_delta_sync_index(
                endpoint_name=VECTOR_SEARCH_ENDPOINT,
                index_name=VS_INDEX_NAME,
                source_table_name=CHUNKS_TABLE,
                pipeline_type="TRIGGERED",
                primary_key="chunk_id",
                embedding_source_column="content",
                embedding_model_endpoint_name=EMBEDDING_MODEL_ENDPOINT,
            )
            return VS_INDEX_NAME
    except Exception as exc:
        logger.warning("Vector search index sync deferred/unavailable (%s)", exc)
        return VS_INDEX_NAME


def index_filing_file(file_path: str) -> int:
    """Orchestrates metadata extraction, deterministic chunking, Delta merge, and VS sync."""
    logger.info("Indexing filing file: %s", file_path)
    meta, chunks = load_and_chunk_file(file_path)
    logger.info(
        "Extracted %s %s (%s, accession: %s) -> %d deterministic chunks.",
        meta["ticker"],
        meta["form_type"],
        meta["year"],
        meta["accession"],
        len(chunks),
    )

    count = write_chunks_to_delta_table(chunks)
    try:
        sync_vector_search_index()
    except Exception as exc:
        logger.debug("VS sync call: %s", exc)

    return count
