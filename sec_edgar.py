"""
Service for fetching and ingesting SEC Edgar filings.
"""
import asyncio
from typing import List, Dict, Optional, Sequence
from datetime import date
from sqlalchemy.ext.asyncio import AsyncSession
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ingestion'))
from ingestion.edgar_fetcher import SecEdgarFetcher, VALID_FORM_TYPES

class SecEdgarService:
    """Service for interacting with SEC EDGAR filings"""

    @staticmethod
    async def list_filings(
        ticker: str,
        form_types: Sequence[str] = VALID_FORM_TYPES,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        output_dir: str = "sec_filings"
    ) -> List[Dict]:
        """
        List available filings for a ticker.
        """
        async with SecEdgarFetcher() as fetcher:
            return await fetcher.list_filings(
                ticker=ticker,
                form_types=form_types,
                start_date=start_date,
                end_date=end_date,
                output_dir=output_dir
            )

    @staticmethod
    async def fetch_filings_stream(
        ticker: str,
        form_types: Sequence[str] = VALID_FORM_TYPES,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        output_dir: str = "sec_filings",
        ingest: bool = True,
        max_concurrent_ingests: int = 3,
        accession_filter: Optional[set] = None,
    ):
        """
        Stream the process of fetching and ingesting filings.
        Yields progress events.
        """
        async with SecEdgarFetcher() as fetcher:
            available = await fetcher.list_filings(ticker, form_types, start_date, end_date, output_dir)
            
            os.makedirs(os.path.join(output_dir, ticker.upper()), exist_ok=True)

            if accession_filter:
                to_fetch = [item for item in available if item["accession"] in accession_filter]
            else:
                to_fetch = available

            queue = asyncio.Queue()
            semaphore = asyncio.Semaphore(max_concurrent_ingests)

            yield {"event": "start", "data": {"total": len(to_fetch), "ticker": ticker}}

            async def _process_one(item: Dict):
                result = {**item, "status": "pending", "error": None, "chunks_added": None}
                
                async with semaphore:
                    try:
                        await queue.put({"event": "progress", "data": {"file": item["form"], "accession": item["accession"], "status": "downloading/rendering"}})

                        # We need to manually duplicate the validation logic here so we can yield events mid-stream
                        # It's cleaner if the fetcher exposes an async generator, but we can just use the fetcher's internals here
                        # or we can rewrite the fetcher to yield, but it's easier to duplicate the loop logic.
                        
                        from ingestion.edgar_fetcher import _is_valid_filing_pdf
                        import logging
                        logger = logging.getLogger("ingestion.edgar_fetcher")

                        if not _is_valid_filing_pdf(item["pdf_path"], item["form"]):
                            if os.path.exists(item["pdf_path"]):
                                logger.warning(
                                    "Existing PDF at %s failed validity check (likely truncated from an "
                                    "earlier interrupted run) — deleting and re-rendering.", item["pdf_path"]
                                )
                                os.remove(item["pdf_path"])
                            await fetcher.convert_url_to_pdf(item["url"], item["pdf_path"], form=item["form"])
                            result["message"] = "Rendered PDF"
                        else:
                            result["message"] = "PDF already exists locally"

                        if ingest:
                            await queue.put({"event": "progress", "data": {"file": item["form"], "accession": item["accession"], "status": "processing/ingesting", "message": result["message"]}})
                            from ingestion.ingest_pdf import ingest_pdf
                            ingest_result = await ingest_pdf(
                                item["pdf_path"], ticker=ticker.upper(), filing_type=item["form"],
                                period_end_date=item.get("period_end_date")
                            )
                            if ingest_result.get("success"):
                                result["success"] = True
                                result["chunks_added"] = ingest_result.get("text_chunks", 0)
                                # Distinguish between new ingestion and already ingested
                                if ingest_result.get("text_processed") or ingest_result.get("images_processed"):
                                    result["status"] = "ingested"
                                else:
                                    result["status"] = "already_ingested"
                            else:
                                result["status"] = "ingest_failed"
                                result["error"] = ingest_result.get("error", "Unknown ingestion error")
                        else:
                            result["status"] = "downloaded"
                            result["success"] = True

                    except Exception as e:
                        result["status"] = "failed"
                        result["error"] = str(e)
                
                await queue.put({"event": "file_completed", "data": result})
                return result

            async def _run_all():
                results = await asyncio.gather(*(_process_one(item) for item in to_fetch))
                
                summary = {
                    "ticker": ticker.upper(),
                    "form_types": list(form_types),
                    "date_range": {"start": str(start_date) if start_date else None, "end": str(end_date) if end_date else None},
                    "total_filings_found": len(to_fetch),
                    "ingested": sum(1 for r in results if r["status"] == "ingested"),
                    "already_ingested": sum(1 for r in results if r["status"] == "already_ingested"),
                    "downloaded_only": sum(1 for r in results if r["status"] == "downloaded"),
                    "failed": sum(1 for r in results if r["status"] in ("failed", "ingest_failed")),
                    "filings": list(results),
                }
                
                await queue.put({"event": "completed", "data": {"results": list(results), "summary": summary}})
                await queue.put(None)

            task = asyncio.create_task(_run_all())

            try:
                while True:
                    event = await queue.get()
                    if event is None:
                        break
                    yield event
            except asyncio.CancelledError:
                task.cancel()
                raise
