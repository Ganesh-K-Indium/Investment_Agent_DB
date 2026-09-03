"""
SEC EDGAR filing fetcher — downloads 10-K / 10-Q / 8-K filings for a ticker,
renders each to PDF, and ingests them into the RAG vector store with the
correct filing_type tag.

Async throughout: HTTP fetches use httpx.AsyncClient, PDF rendering uses
Playwright's async API, and ingestion itself (pdf_processor1.py) is async —
embeddings via OpenAI's async client, upserts via Qdrant's AsyncQdrantClient.
Filings ingest concurrently (bounded by a semaphore); PDF text/OCR extraction
is offloaded to a worker thread (see pdf_processor1.py) so it doesn't block
the event loop while other requests are in flight.

Each filing's exact period_end_date is taken from EDGAR's own `reportDate`
field (SEC ground truth, not re-derived from the document) and passed to
ingest_pdf() as an explicit override.

Supersedes the old root-level download_10q_pdfs.py (10-Q-only, hardcoded to
AAPL/2025, no ingestion hookup).
"""
import asyncio
import logging
import os
import time
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence

import httpx
from dotenv import load_dotenv
from playwright.async_api import async_playwright

load_dotenv()

logger = logging.getLogger("ingestion.edgar_fetcher")

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "Indium Capital contact@indium.com")
SEC_REQUEST_RATE_LIMIT = 10  # max 10 requests/sec (SEC guideline) — enforced per-IP, not per-instance

VALID_FORM_TYPES = ("10-K", "10-Q", "8-K")

# Minimum plausible page count per filing type — a sanity floor, not an exact
# expectation. 10-K/10-Q are always substantial (audited/condensed financial
# statements + footnotes + MD&A); 8-Ks are genuinely often just 1-3 pages, so
# they get a much lower floor. Anything under this for its form type almost
# certainly means a truncated/interrupted render, not a real short filing.
_MIN_PLAUSIBLE_PAGES = {"10-K": 15, "10-Q": 10, "8-K": 1}


def _is_valid_filing_pdf(pdf_path: str, form: Optional[str] = None) -> bool:
    """
    Sanity-check a rendered/downloaded filing PDF: does it open cleanly, and
    does its page count clear a minimum plausible floor for its form type?
    Used both right after rendering (fail fast) and before trusting an
    already-on-disk file (catch a truncated file left by a prior interrupted
    run, which `os.path.exists()` alone can't distinguish from a good one).
    """
    if not os.path.exists(pdf_path):
        return False
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
    except Exception:
        return False
    return page_count >= _MIN_PLAUSIBLE_PAGES.get(form, 1)


class _SharedRateLimiter:
    """
    Process-wide rate limiter shared across every SecEdgarFetcher instance.

    SEC's 10 req/sec guideline is per source IP, not per process/instance.
    Tracking `_last_request_time` on `self` (the old approach) gave each
    concurrent SecEdgarFetcher() — e.g. two /edgar/ingest requests for
    different tickers in flight at once — its own independent budget, so N
    concurrent ingestions could collectively exceed the real limit and risk
    SEC throttling/blocking the whole server's IP. A module-level lock +
    timestamp, awaited by every instance, keeps the true combined request
    rate under the limit regardless of how many fetchers are active.
    """

    def __init__(self, rate_limit: float):
        self._delay = 1.0 / rate_limit
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self._delay:
                await asyncio.sleep(self._delay - elapsed)
            self._last_request_time = time.monotonic()


_edgar_rate_limiter = _SharedRateLimiter(SEC_REQUEST_RATE_LIMIT)


class SecEdgarFetcher:
    """Fetches SEC filings (10-K/10-Q/8-K), converts them to PDF, and ingests them."""

    def __init__(self):
        self.headers = {
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
            "Connection": "keep-alive",
        }
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _rate_limited_request(self, url: str) -> httpx.Response:
        """Ensure SEC rate limits are respected, across ALL concurrent fetchers."""
        await _edgar_rate_limiter.wait()

        response = await self._client.get(url, headers=self.headers)
        response.raise_for_status()
        return response

    async def get_cik(self, ticker: str) -> str:
        """Convert ticker to CIK."""
        url = "https://www.sec.gov/files/company_tickers.json"
        data = (await self._rate_limited_request(url)).json()

        for entry in data.values():
            if entry["ticker"].lower() == ticker.lower():
                return str(entry["cik_str"]).zfill(10)

        raise ValueError(f"CIK not found for ticker {ticker}")

    async def convert_url_to_pdf(self, url: str, output_path: str, form: Optional[str] = None):
        """Render a SEC HTML filing page to PDF.

        SEC.gov's WAF flags headless Chromium *navigating directly* to
        sec.gov as an "undeclared automated tool" and serves a block page —
        even with a declared User-Agent — while a plain httpx GET with the
        same UA succeeds fine. So: fetch the HTML via httpx (already proven
        to work, same client used for the JSON API calls), then hand that
        content to Playwright via set_content() purely for local rendering —
        the browser never makes a request to sec.gov itself, sidestepping
        the block entirely.
        """
        html_response = await self._rate_limited_request(url)
        html_content = html_response.text

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                context = await browser.new_context(user_agent=SEC_USER_AGENT)
                page = await context.new_page()
                # Sub-resources (images/CSS) would still hit sec.gov and could
                # trip the same block — not needed for text extraction, so drop them.
                await page.route("**/*", lambda route: (
                    route.continue_() if route.request.url == "about:blank" or route.request.resource_type == "document"
                    else route.abort()
                ))
                await page.set_content(html_content, wait_until="domcontentloaded", timeout=30000)
                await page.pdf(path=output_path, format="A4", print_background=True)
            finally:
                await browser.close()

        # A render that got interrupted partway (killed process, network drop
        # mid-fetch of html_content, etc.) can still leave a syntactically
        # valid, non-empty PDF on disk — just a truncated one (e.g. an 8-page
        # file for a filing that should be 100+ pages). Every later run then
        # sees the file already exists and skips re-rendering forever,
        # silently ingesting (or failing to ingest) garbage. Validate the
        # output immediately and delete+raise rather than leaving a bad file
        # behind for a future run to trip over.
        if not _is_valid_filing_pdf(output_path, form):
            if os.path.exists(output_path):
                os.remove(output_path)
            raise RuntimeError(
                f"Rendered PDF for {output_path} looks incomplete/corrupt "
                f"(failed page-count sanity check for form type {form!r}) — deleted, not left on disk."
            )

    async def list_filings(
        self,
        ticker: str,
        form_types: Sequence[str] = VALID_FORM_TYPES,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        output_dir: str = "sec_filings",
    ) -> List[Dict]:
        """
        List available filings for a ticker WITHOUT downloading/ingesting anything —
        metadata only (form type, filing date, period-end date, accession number,
        the URL and local pdf_path it would use). Used to let a caller (e.g. an
        interactive CLI) show what's available and pick specific ones before any
        network-heavy work happens.

        Args:
            ticker: Stock ticker symbol.
            form_types: SEC form types to include (subset of "10-K", "10-Q", "8-K").
            start_date: Only include filings on/after this date. Defaults to no lower bound.
            end_date: Only include filings on/before this date. Defaults to today.
            output_dir: Directory that would hold rendered PDFs (used to build pdf_path).

        Returns:
            List of dicts, most recent first: {form, filing_date, period_end_date,
            accession, url, pdf_path}.
        """
        invalid = [f for f in form_types if f not in VALID_FORM_TYPES]
        if invalid:
            raise ValueError(f"Invalid form_types {invalid}. Must be a subset of {VALID_FORM_TYPES}.")

        end_date = end_date or date.today()

        cik = await self.get_cik(ticker)

        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = (await self._rate_limited_request(url)).json()

        ticker_dir = os.path.join(output_dir, ticker.upper())

        def _parse_block(filings: dict) -> list:
            block = []
            for i in range(len(filings["form"])):
                form = filings["form"][i]
                if form not in form_types:
                    continue

                filing_date_str = filings["filingDate"][i]
                filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").date()
                if start_date and filing_date < start_date:
                    continue
                if filing_date > end_date:
                    continue

                accession_raw = filings["accessionNumber"][i]
                accession = accession_raw.replace("-", "")
                primary_doc = filings["primaryDocument"][i]

                # EDGAR's `reportDate` is the actual period this filing covers (fiscal
                # year end for a 10-K, fiscal quarter end for a 10-Q, event date for an
                # 8-K) — distinct from `filingDate` (when it was submitted, typically
                # weeks later). This is SEC ground truth, so it's passed to ingest_pdf()
                # as an explicit override — it always wins over cover-page/filename
                # detection rather than being re-derived.
                report_date_str = filings.get("reportDate", [None] * len(filings["form"]))[i] or None

                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{accession}/{primary_doc}"
                )
                pdf_path = os.path.join(ticker_dir, f"{ticker.upper()}_{form.replace('/', '-')}_{filing_date_str}_{accession}.pdf")

                block.append({
                    "form": form,
                    "filing_date": filing_date_str,
                    "period_end_date": report_date_str,
                    "accession": accession,
                    "url": filing_url,
                    "pdf_path": pdf_path
                })
            return block

        available = _parse_block(data["filings"]["recent"])

        # The "recent" block above is capped at ~1000 entries total across
        # EVERY form type (10-K/10-Q/8-K plus Form 4/144/13F/etc.) — for a
        # ticker with heavy insider-transaction filing volume (e.g. a
        # megacap with frequent Form 4/144s), that cap can be exhausted by
        # non-10-K/10-Q/8-K noise within just the last 1-2 years, silently
        # pushing OLDER 10-Ks/10-Qs/8-Ks out of view even though they still
        # exist on EDGAR. Older history lives in separate paginated files
        # listed under filings.files — fetch whichever of those overlap the
        # requested date range (all of them, when start_date is None, since
        # that means "no lower bound" i.e. full history).
        for file_meta in data["filings"].get("files", []):
            page_from = datetime.strptime(file_meta["filingFrom"], "%Y-%m-%d").date()
            page_to = datetime.strptime(file_meta["filingTo"], "%Y-%m-%d").date()
            if start_date and page_to < start_date:
                continue
            if page_from > end_date:
                continue

            page_url = f"https://data.sec.gov/submissions/{file_meta['name']}"
            page_data = (await self._rate_limited_request(page_url)).json()
            available.extend(_parse_block(page_data))

        # Pages arrive oldest-chunk-last but each chunk is itself already in
        # EDGAR's native (most-recent-first) order — re-sort the merged list
        # so callers always get a single consistent most-recent-first view.
        available.sort(key=lambda f: f["filing_date"], reverse=True)

        return available

    async def fetch_filings(
        self,
        ticker: str,
        form_types: Sequence[str] = VALID_FORM_TYPES,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        output_dir: str = "sec_filings",
        ingest: bool = True,
        max_concurrent_ingests: int = 3,
        accession_filter: Optional[set] = None,
    ) -> Dict:
        """
        Fetch, download, and (optionally) ingest filings for a ticker.

        Args:
            ticker: Stock ticker symbol.
            form_types: SEC form types to fetch (subset of "10-K", "10-Q", "8-K").
            start_date: Only include filings on/after this date. Defaults to no lower bound.
            end_date: Only include filings on/before this date. Defaults to today.
            output_dir: Directory to save rendered PDFs under (one subfolder per ticker).
            ingest: If True, ingest each downloaded PDF into the vector store.
            max_concurrent_ingests: Bound on concurrent ingestion (embedding/Qdrant calls).
            accession_filter: If provided, only process filings whose accession number
                (no dashes) is in this set — use with list_filings() to let a caller
                fetch/ingest a specific hand-picked subset instead of everything
                matching form_types/date range.

        Returns:
            Summary dict with per-filing results and aggregate counts.
        """
        available = await self.list_filings(ticker, form_types, start_date, end_date, output_dir)

        os.makedirs(os.path.join(output_dir, ticker.upper()), exist_ok=True)

        if accession_filter:
            to_fetch = [item for item in available if item["accession"] in accession_filter]
        else:
            to_fetch = available

        results = []
        semaphore = asyncio.Semaphore(max_concurrent_ingests)

        async def _process_one(item: Dict) -> Dict:
            result = {**item, "status": "pending", "error": None, "chunks_added": None}
            try:
                if not _is_valid_filing_pdf(item["pdf_path"], item["form"]):
                    if os.path.exists(item["pdf_path"]):
                        logger.warning(
                            "Existing PDF at %s failed validity check (likely truncated from an "
                            "earlier interrupted run) — deleting and re-rendering.", item["pdf_path"]
                        )
                        os.remove(item["pdf_path"])
                    await self.convert_url_to_pdf(item["url"], item["pdf_path"], form=item["form"])
                    result["message"] = "Rendered PDF"
                else:
                    result["message"] = "PDF already exists locally"

                if ingest:
                    async with semaphore:
                        from ingestion.ingest_pdf import ingest_pdf
                        ingest_result = await ingest_pdf(
                            item["pdf_path"], ticker=ticker.upper(), filing_type=item["form"],
                            period_end_date=item.get("period_end_date")
                        )
                    if ingest_result.get("success"):
                        result["status"] = "ingested"
                        result["chunks_added"] = ingest_result.get("text_chunks", 0)
                    else:
                        result["status"] = "ingest_failed"
                        result["error"] = ingest_result.get("error", "Unknown ingestion error")
                else:
                    result["status"] = "downloaded"

            except Exception as e:
                result["status"] = "failed"
                result["error"] = str(e)

            return result

        results = await asyncio.gather(*(_process_one(item) for item in to_fetch))

        summary = {
            "ticker": ticker.upper(),
            "form_types": list(form_types),
            "date_range": {"start": str(start_date) if start_date else None, "end": str(end_date)},
            "total_filings_found": len(to_fetch),
            "ingested": sum(1 for r in results if r["status"] == "ingested"),
            "downloaded_only": sum(1 for r in results if r["status"] == "downloaded"),
            "failed": sum(1 for r in results if r["status"] in ("failed", "ingest_failed")),
            "filings": list(results),
        }
        return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and ingest SEC filings for a ticker.")
    parser.add_argument("ticker")
    parser.add_argument("--form-types", nargs="+", default=list(VALID_FORM_TYPES), choices=VALID_FORM_TYPES)
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--no-ingest", action="store_true", help="Download PDFs only, skip vector ingestion.")
    parser.add_argument("--output-dir", default="sec_filings")
    args = parser.parse_args()

    async def _main():
        async with SecEdgarFetcher() as fetcher:
            summary = await fetcher.fetch_filings(
                ticker=args.ticker,
                form_types=args.form_types,
                start_date=args.start_date,
                end_date=args.end_date,
                output_dir=args.output_dir,
                ingest=not args.no_ingest,
            )
            logger.info(summary)

    asyncio.run(_main())
