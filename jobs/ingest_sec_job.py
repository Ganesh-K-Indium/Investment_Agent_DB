"""
Standalone and Serverless Databricks Job Entrypoint for SEC Ingestion.
Supports:
1. Discovery filtering: --ticker, --forms, --year, --quarter, --start-date, --end-date.
2. Granular accession targeting: --accessions to selectively ingest specific filings.
3. Discovery mode: --discover-only to list filings and check indexing status without downloading.
4. Error recovery: Isolates failures per filing and provides full visibility into completed tasks.
"""

import argparse
import sys
import os
import json
import logging
from typing import List, Optional, Dict

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA,
    DATABRICKS_VOLUME,
    CHUNKS_TABLE,
    VS_INDEX_NAME,
)
from data_pipeline.sec_loader import discover_filings_sync, load_specific_filing_sync
from data_pipeline.vector_indexer import index_filing_file, sync_vector_search_index
from tools.uc_tools import check_multiple_accessions_status

logger = logging.getLogger("jobs.ingest_sec_job")


def run_batch_ingestion(
    ticker: str,
    form_types: Optional[List[str]] = None,
    year: Optional[int] = None,
    quarter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_accessions: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    discover_only: bool = False,
) -> Dict:
    """Discovers, filters, and selectively ingests SEC filings with deduplication and error recovery."""
    clean_ticker = ticker.strip().upper()
    logger.info("=== SEC Ingestion Pipeline: %s ===", clean_ticker)
    
    # 1. Discovery Phase
    discovered = discover_filings_sync(
        ticker=clean_ticker,
        form_types=form_types,
        year=year,
        quarter=quarter,
        start_date=start_date,
        end_date=end_date,
    )
    logger.info("Discovered %d candidate filings from SEC EDGAR.", len(discovered))

    # Check existing index status
    accession_list = [f["accession"] for f in discovered]
    status_map = check_multiple_accessions_status(clean_ticker, accession_list)
    for f in discovered:
        f["indexed_chunks"] = status_map.get(f["accession"], 0)
        f["is_indexed"] = f["indexed_chunks"] > 0

    if discover_only:
        return {
            "status": "DISCOVERY_ONLY",
            "ticker": clean_ticker,
            "total_found": len(discovered),
            "filings": discovered,
        }

    # 2. Filter target filings if specific accessions requested
    if target_accessions:
        clean_target_accs = {a.replace("-", "").strip() for a in target_accessions}
        to_process = [f for f in discovered if f["accession"] in clean_target_accs]
    else:
        to_process = discovered

    if not to_process:
        logger.warning("No filings matched criteria for ingestion.")
        return {
            "status": "NO_MATCH",
            "ticker": clean_ticker,
            "total_found": len(discovered),
            "processed": 0,
            "results": [],
        }

    logger.info("Starting selective ingestion for %d filings...", len(to_process))
    results = []
    total_chunks_added = 0

    for idx, filing in enumerate(to_process, 1):
        f_desc = f"{filing['ticker']} {filing['form']} ({filing['filing_date']}, Acc: {filing['accession']})"
        logger.info("[%d/%d] Ingesting %s...", idx, len(to_process), f_desc)
        try:
            # Download and clean text to UC Volume
            saved_path = load_specific_filing_sync(filing, output_dir=output_dir)
            
            # Chunk and merge to Delta with CDF
            chunks_indexed = index_filing_file(saved_path)
            total_chunks_added += chunks_indexed

            results.append({
                "accession": filing["accession"],
                "form": filing["form"],
                "year": filing["year"],
                "quarter": filing["quarter"],
                "status": "SUCCESS",
                "chunks_indexed": chunks_indexed,
                "file_path": saved_path,
                "error": None,
            })
            logger.info("Successfully indexed %s (%d chunks)", f_desc, chunks_indexed)
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", f_desc, exc)
            results.append({
                "accession": filing["accession"],
                "form": filing["form"],
                "year": filing["year"],
                "quarter": filing["quarter"],
                "status": "FAILED",
                "chunks_indexed": 0,
                "file_path": None,
                "error": str(exc),
            })

    # Trigger vector search index sync once after batch
    try:
        sync_vector_search_index()
    except Exception as e:
        logger.warning("Vector search sync notification: %s", e)

    successful = sum(1 for r in results if r["status"] == "SUCCESS")
    failed = sum(1 for r in results if r["status"] == "FAILED")

    summary = {
        "status": "COMPLETED" if failed == 0 else "PARTIAL_SUCCESS",
        "ticker": clean_ticker,
        "total_attempted": len(to_process),
        "successful": successful,
        "failed": failed,
        "total_chunks_indexed": total_chunks_added,
        "delta_table": CHUNKS_TABLE,
        "vector_search_index": VS_INDEX_NAME,
        "results": results,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="SEC Edgar Granular Ingestion & Delta Indexing Job")
    parser.add_argument("--ticker", required=True, type=str, help="Company stock ticker (e.g. NVDA, AAPL, MSFT)")
    parser.add_argument("--forms", nargs="+", choices=["10-K", "10-Q", "8-K"], default=None, help="SEC Form types to filter")
    parser.add_argument("--form", type=str, choices=["10-K", "10-Q", "8-K"], default=None, help="Single form (backwards compatible)")
    parser.add_argument("--year", type=int, default=None, help="Fiscal or filing year (e.g. 2024)")
    parser.add_argument("--quarter", type=str, choices=["Q1", "Q2", "Q3", "Q4"], default=None, help="Fiscal quarter")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--accessions", nargs="+", default=None, help="Specific accession numbers to ingest")
    parser.add_argument("--output-dir", type=str, default=None, help="Target storage directory override")
    parser.add_argument("--discover-only", action="store_true", help="Only discover and print metadata without ingesting")

    args = parser.parse_args()

    form_types = args.forms
    if not form_types and args.form:
        form_types = [args.form]

    try:
        summary = run_batch_ingestion(
            ticker=args.ticker,
            form_types=form_types,
            year=args.year,
            quarter=args.quarter,
            start_date=args.start_date,
            end_date=args.end_date,
            target_accessions=args.accessions,
            output_dir=args.output_dir,
            discover_only=args.discover_only,
        )
        print(json.dumps(summary, indent=2))
        sys.exit(0 if summary.get("failed", 0) == 0 else 1)
    except Exception as exc:
        logger.exception("Fatal job error: %s", exc)
        print(f"[FATAL ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
