"""
Unity Catalog Governed Tools for SEC Intelligence.
Implements filing status checks, HITL feedback persistence, and memory retrieval.
Designed for registration as governed Unity Catalog functions and MCP-ready tools.
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from config import CHUNKS_TABLE, FEEDBACK_TABLE

logger = logging.getLogger(__name__)


def _get_workspace_client_and_warehouse():
    """Helper to lazily initialize WorkspaceClient and locate an active SQL Warehouse."""
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No active Databricks SQL Warehouse found for tool execution.")
    return w, warehouses[0].id


def check_accession_status(ticker: str, accession: str) -> str:
    """Checks if a specific SEC filing by accession number exists in the Delta index table.

    Args:
        ticker: Stock ticker symbol (e.g., 'NVDA', 'AAPL').
        accession: SEC accession number (without dashes).

    Returns:
        JSON string with 'status' ('INDEXED' or 'NOT_FOUND') and 'chunk_count'.
    """
    clean_ticker = ticker.strip().upper()
    clean_acc = accession.replace("-", "").strip()

    try:
        w, warehouse_id = _get_workspace_client_and_warehouse()
        query = f"""
        SELECT COUNT(*) AS chunk_count
        FROM {CHUNKS_TABLE}
        WHERE UPPER(ticker) = '{clean_ticker}'
          AND accession = '{clean_acc}';
        """
        response = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=query,
            wait_timeout="20s",
        )
        chunk_count = 0
        if response.result and response.result.data_array:
            chunk_count = int(response.result.data_array[0][0])
        status = "INDEXED" if chunk_count > 0 else "NOT_FOUND"
        return json.dumps({"status": status, "chunk_count": chunk_count, "accession": clean_acc, "ticker": clean_ticker})
    except Exception as exc:
        return json.dumps({"status": "NOT_FOUND", "chunk_count": 0, "accession": clean_acc, "ticker": clean_ticker, "detail": str(exc)})


def check_multiple_accessions_status(ticker: str, accessions: list) -> dict:
    """Batch checks indexing status for a list of accession numbers.

    Returns:
        Dictionary mapping accession -> chunk_count.
    """
    clean_ticker = ticker.strip().upper()
    clean_accs = [a.replace("-", "").strip() for a in accessions if a]
    results = {a: 0 for a in clean_accs}

    if not clean_accs:
        return results

    try:
        w, warehouse_id = _get_workspace_client_and_warehouse()
        in_list = ", ".join(f"'{a}'" for a in clean_accs)
        query = f"""
        SELECT accession, COUNT(*) AS chunk_count
        FROM {CHUNKS_TABLE}
        WHERE UPPER(ticker) = '{clean_ticker}'
          AND accession IN ({in_list})
        GROUP BY accession;
        """
        response = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=query,
            wait_timeout="25s",
        )
        if response.result and response.result.data_array:
            for row in response.result.data_array:
                acc_val = str(row[0])
                count_val = int(row[1])
                results[acc_val] = count_val
    except Exception as exc:
        logger.debug("Batch accession status check: %s", exc)

    return results



def check_filing_status(ticker: str, form_type: str, year: int) -> str:
    """Checks if chunks for the target SEC filing exist in the Unity Catalog Delta table.

    Queries {CATALOG}.{SCHEMA}.sec_filing_chunks to verify whether the filing
    has already been ingested, chunked, and prepared for vector search.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT').
        form_type: SEC Form type (e.g., '10-K', '10-Q', '8-K').
        year: Fiscal or filing year (e.g., 2023, 2024).

    Returns:
        A JSON string containing:
        - status: 'INDEXED' if chunks exist, otherwise 'NOT_FOUND'
        - chunk_count: Number of chunks found in the Delta table
        - ticker: Normalized uppercase ticker
        - form_type: Normalized uppercase form type
        - year: The fiscal/filing year queried
    """
    clean_ticker = ticker.strip().upper()
    clean_form = form_type.strip().upper()
    clean_year = int(year)

    try:
        w, warehouse_id = _get_workspace_client_and_warehouse()
        query = f"""
        SELECT COUNT(*) AS chunk_count
        FROM {CHUNKS_TABLE}
        WHERE UPPER(ticker) = '{clean_ticker}'
          AND UPPER(form_type) = '{clean_form}'
          AND year = {clean_year};
        """
        response = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=query,
            wait_timeout="30s",
        )
        
        chunk_count = 0
        if response.result and response.result.data_array:
            chunk_count = int(response.result.data_array[0][0])

        status = "INDEXED" if chunk_count > 0 else "NOT_FOUND"
        return json.dumps({
            "status": status,
            "chunk_count": chunk_count,
            "ticker": clean_ticker,
            "form_type": clean_form,
            "year": clean_year,
        })
    except Exception as exc:
        logger.warning("Filing status check failed against UC (%s). Returning NOT_FOUND.", exc)
        return json.dumps({
            "status": "NOT_FOUND",
            "chunk_count": 0,
            "ticker": clean_ticker,
            "form_type": clean_form,
            "year": clean_year,
            "detail": str(exc),
        })


def record_feedback(
    query: str,
    ticker: str,
    rating: str,
    feedback_text: str,
    corrected_context: str,
) -> str:
    """Inserts a human-in-the-loop (HITL) feedback row into the Delta memory table.

    Persists user ratings, critiques, corrections, and explicit instructions into
    {CATALOG}.{SCHEMA}.agent_feedback so future agent reasoning loops can adapt.

    Args:
        query: The user prompt or investment question being reviewed.
        ticker: The stock ticker symbol associated with the analysis.
        rating: Feedback rating (e.g. 'POSITIVE', 'NEGATIVE', 'THUMBS_UP', 'THUMBS_DOWN').
        feedback_text: Detailed critique, correction, or preference from the user.
        corrected_context: Optional text provided by the user containing the corrected ground truth.

    Returns:
        A JSON string containing the status and the generated feedback_id.
    """
    feedback_id = str(uuid.uuid4())
    clean_ticker = ticker.strip().upper()
    now_iso = datetime.now(timezone.utc).isoformat()

    # Escape strings for SQL statement
    def _escape(val: str) -> str:
        return val.replace("'", "''").replace("\\", "\\\\")

    safe_query = _escape(query)
    safe_rating = _escape(rating.strip().upper())
    safe_feedback = _escape(feedback_text)
    safe_corrected = _escape(corrected_context)

    try:
        w, warehouse_id = _get_workspace_client_and_warehouse()

        insert_sql = f"""
        INSERT INTO {FEEDBACK_TABLE}
        (feedback_id, timestamp, ticker, query, rating, feedback_text, corrected_context)
        VALUES (
            '{feedback_id}',
            current_timestamp(),
            '{clean_ticker}',
            '{safe_query}',
            '{safe_rating}',
            '{safe_feedback}',
            '{safe_corrected}'
        );
        """
        w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=insert_sql,
            wait_timeout="30s",
        )
        logger.info("Recorded feedback %s for ticker %s in %s", feedback_id, clean_ticker, FEEDBACK_TABLE)
        return json.dumps({
            "status": "SUCCESS",
            "feedback_id": feedback_id,
            "ticker": clean_ticker,
            "timestamp": now_iso,
        })
    except Exception as exc:
        logger.error("Failed to insert feedback row into %s: %s", FEEDBACK_TABLE, exc)
        return json.dumps({
            "status": "ERROR",
            "error": str(exc),
            "feedback_id": feedback_id,
            "ticker": clean_ticker,
        })


def get_relevant_feedback(ticker: str, query_topic: str) -> str:
    """Retrieves previous user critiques and corrections from the Delta memory table.

    Searches {CATALOG}.{SCHEMA}.agent_feedback for negative ratings or specific user
    instructions associated with this company ticker or analytical topic.

    Args:
        ticker: The stock ticker symbol to look up.
        query_topic: Analytical topic or query snippet (e.g. 'gross margins', 'risk factors').

    Returns:
        A formatted string summarizing past user corrections and preferences, or an advisory
        that no prior feedback was found.
    """
    clean_ticker = ticker.strip().upper()
    try:
        w, warehouse_id = _get_workspace_client_and_warehouse()
        select_sql = f"""
        SELECT timestamp, rating, feedback_text, corrected_context
        FROM {FEEDBACK_TABLE}
        WHERE UPPER(ticker) = '{clean_ticker}'
        ORDER BY timestamp DESC
        LIMIT 5;
        """
        response = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=select_sql,
            wait_timeout="30s",
        )

        rows = []
        if response.result and response.result.data_array:
            rows = response.result.data_array

        if not rows:
            return f"No past user critiques or corrections recorded for ticker {clean_ticker}."

        critiques = []
        for idx, row in enumerate(rows, 1):
            ts, rating, feedback, corrected = row[0], row[1], row[2], row[3]
            entry = f"- [Entry {idx} | {rating} | {ts}]: {feedback}"
            if corrected:
                entry += f" (Corrected Ground Truth: '{corrected}')"
            critiques.append(entry)

        summary = (
            f"### Past User Feedback & Guidelines for {clean_ticker}:\n"
            + "\n".join(critiques)
            + "\n\n*Apply these user corrections and preferences to your current analytical plan and synthesis.*"
        )
        return summary
    except Exception as exc:
        logger.warning("Feedback retrieval from %s failed (%s). Returning default response.", FEEDBACK_TABLE, exc)
        return f"No past user critiques or corrections retrieved (memory unavailable: {exc})."

