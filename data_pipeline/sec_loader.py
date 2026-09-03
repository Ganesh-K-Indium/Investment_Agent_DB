"""
SEC EDGAR Loader & Parser with Granular Discovery and Filtering.
Supports:
1. Discovery: Filter filings by form types (10-K, 10-Q, 8-K), fiscal year, quarter (Q1-Q4), or custom date range.
2. Selective Ingestion: Download, parse, and persist specific user-chosen filings.
3. UC Volume Integration: Saves clean text directly to /Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/.
"""

import asyncio
import os
import re
import time
import logging
from datetime import datetime, date
from typing import Dict, List, Optional, Sequence, Union
import httpx
from bs4 import BeautifulSoup

from config import (
    SEC_USER_AGENT,
    SEC_REQUEST_RATE_LIMIT,
    VOLUME_PATH,
    LOCAL_FALLBACK_VOLUME,
    DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA,
    DATABRICKS_VOLUME,
)

logger = logging.getLogger(__name__)

VALID_FORM_TYPES = ("10-K", "10-Q", "8-K")


class _SharedRateLimiter:
    """Process-wide rate limiter respecting SEC's 10 req/sec guideline."""

    def __init__(self, rate_limit: float = SEC_REQUEST_RATE_LIMIT):
        self._delay = 1.0 / max(rate_limit, 1.0)
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def wait(self):
        async with self._lock:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self._delay:
                await asyncio.sleep(self._delay - elapsed)
            self._last_request_time = time.monotonic()


_rate_limiter = _SharedRateLimiter(SEC_REQUEST_RATE_LIMIT)


def get_target_volume_dir(override_path: Optional[str] = None) -> str:
    """
    Resolve the target storage directory.
    Prefers Unity Catalog Volume path `/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/`.
    Falls back to a local directory if UC volume path is inaccessible.
    """
    if override_path:
        os.makedirs(override_path, exist_ok=True)
        return override_path

    try:
        os.makedirs(VOLUME_PATH, exist_ok=True)
        return VOLUME_PATH
    except (OSError, PermissionError) as exc:
        logger.warning(
            "Cannot write to UC Volume path '%s' (%s). Using local fallback: '%s'",
            VOLUME_PATH,
            exc,
            LOCAL_FALLBACK_VOLUME,
        )
        os.makedirs(LOCAL_FALLBACK_VOLUME, exist_ok=True)
        return LOCAL_FALLBACK_VOLUME


def clean_html_to_text(html_content: str) -> str:
    """
    Parse and clean SEC filing HTML or raw text into readable markdown/plain text.
    Removes scripts, styles, XML artifacts, and collapses whitespace.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script, style, and metadata elements
    for element in soup(["script", "style", "head", "title", "meta", "[document]"]):
        element.extract()

    # Extract text with smart line breaks
    lines = []
    for element in soup.descendants:
        if element.name in ["p", "div", "h1", "h2", "h3", "h4", "tr", "li", "br"]:
            lines.append("\n")
        elif element.string and not element.name:
            text = element.string.strip()
            if text:
                lines.append(f" {text} ")

    raw_text = "".join(lines)
    raw_text = re.sub(r"[ \t]+", " ", raw_text)
    raw_text = re.sub(r"\n\s*\n+", "\n\n", raw_text)
    return raw_text.strip()


def compute_quarter(date_obj: date) -> str:
    """Computes calendar quarter string (e.g. Q1, Q2, Q3, Q4)."""
    q_num = (date_obj.month - 1) // 3 + 1
    return f"Q{q_num}"


def compute_fiscal_period_and_year(
    report_date: date,
    form_type: str,
    fiscal_year_end_mmdd: Optional[str] = "1231",
) -> tuple[str, int]:
    """
    Computes exact corporate fiscal quarter (Q1, Q2, Q3, Q4 / FY) and fiscal year
    handling arbitrary corporate fiscal year ends (e.g. NVDA Jan 31, AAPL Sep 30, MSFT Jun 30).

    Args:
        report_date: Balance sheet / period end date from SEC EDGAR.
        form_type: SEC form type ('10-K', '10-Q', '8-K').
        fiscal_year_end_mmdd: 4-character MMDD string from SEC EDGAR (e.g. '0131' for Jan 31).

    Returns:
        Tuple of (fiscal_period, fiscal_year) e.g. ('Q1', 2025) or ('FY', 2024).
    """
    if not fiscal_year_end_mmdd or len(fiscal_year_end_mmdd) < 2:
        fiscal_year_end_mmdd = "1231"

    try:
        fye_month = int(fiscal_year_end_mmdd[:2])
    except ValueError:
        fye_month = 12

    # Calendar standard year-end (e.g. Dec 31)
    if fye_month == 12:
        q_num = (report_date.month - 1) // 3 + 1
        period_str = "FY" if form_type.upper() == "10-K" else f"Q{q_num}"
        return period_str, report_date.year

    # Off-calendar fiscal year (e.g. NVDA ends Jan, AAPL ends Sep, MSFT ends Jun)
    months_after_fye = (report_date.month - fye_month) % 12
    quarter_num = 4 if months_after_fye == 0 else (months_after_fye - 1) // 3 + 1

    # In corporate reporting, periods after fiscal year end month M belong to the NEXT fiscal year
    if report_date.month > fye_month:
        fiscal_year = report_date.year + 1
    else:
        fiscal_year = report_date.year

    if form_type.upper() == "10-K":
        return "FY", fiscal_year

    return f"Q{quarter_num}", fiscal_year


class SECLoader:
    """Databricks-native SEC EDGAR filing loader with discovery and selective filtering."""

    def __init__(self, user_agent: str = SEC_USER_AGENT):
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json, text/html, */*",
            "Connection": "keep-alive",
        }
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=45.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _request(self, url: str) -> httpx.Response:
        """Perform a rate-limited HTTP GET request with standard SEC headers."""
        await _rate_limiter.wait()
        if not self._client:
            self._client = httpx.AsyncClient(timeout=45.0)
        resp = await self._client.get(url, headers=self.headers)
        resp.raise_for_status()
        return resp

    async def get_cik(self, ticker: str) -> str:
        """Resolve ticker symbol to 10-digit zero-padded CIK string."""
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = await self._request(url)
        data = resp.json()
        ticker_lower = ticker.lower()
        for entry in data.values():
            if entry.get("ticker", "").lower() == ticker_lower:
                return str(entry["cik_str"]).zfill(10)
        raise ValueError(f"Ticker '{ticker}' not found in SEC CIK ticker registry.")

    async def discover_filings(
        self,
        ticker: str,
        form_types: Optional[Sequence[str]] = None,
        year: Optional[int] = None,
        quarter: Optional[str] = None,
        start_date: Optional[Union[date, str]] = None,
        end_date: Optional[Union[date, str]] = None,
        year_type: str = "fiscal",
    ) -> List[Dict]:
        """
        Discovers all available filings for a ticker based on user criteria without downloading full text.
        
        Args:
            ticker: Stock ticker symbol (e.g. AAPL, NVDA).
            form_types: Subset of ('10-K', '10-Q', '8-K').
            year: Target year to filter by.
            quarter: Target quarter ('Q1', 'Q2', 'Q3'). 10-Ks are excluded if Q1/Q2/Q3 is specified.
            start_date: Optional lower date bound (inclusive).
            end_date: Optional upper date bound (inclusive).
            year_type: 'fiscal' (default, matches company's fiscal year) or 'calendar' (matches filing submission year).
        """
        allowed_forms = set(f.upper() for f in (form_types or VALID_FORM_TYPES))
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date() if isinstance(start_date, str) else start_date
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date() if isinstance(end_date, str) else end_date

        cik = await self.get_cik(ticker)
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        data = (await self._request(url)).json()

        fye_mmdd = data.get("fiscalYearEnd", "1231") or "1231"
        formatted_fye = f"{fye_mmdd[:2]}/{fye_mmdd[2:]}" if len(fye_mmdd) >= 4 else "12/31"

        def _parse_filings_block(filings_meta: dict) -> List[Dict]:
            extracted = []
            forms = filings_meta.get("form", [])
            for i in range(len(forms)):
                form = forms[i].upper()
                if form not in allowed_forms:
                    continue

                filing_date_str = filings_meta["filingDate"][i]
                filing_dt = datetime.strptime(filing_date_str, "%Y-%m-%d").date()

                if start_dt and filing_dt < start_dt:
                    continue
                if end_dt and filing_dt > end_dt:
                    continue

                report_date_str = filings_meta.get("reportDate", [None] * len(forms))[i] or filing_date_str
                report_dt = datetime.strptime(report_date_str, "%Y-%m-%d").date() if report_date_str else filing_dt

                # Compute official corporate fiscal period and fiscal year
                fiscal_period, fiscal_year = compute_fiscal_period_and_year(report_dt, form, fye_mmdd)
                calendar_quarter = compute_quarter(report_dt) if form == "10-Q" else "N/A"

                # Strict year matching (default is fiscal year; calendar if specified)
                if year:
                    target_y = int(year)
                    if year_type == "fiscal":
                        if fiscal_year != target_y:
                            continue
                    else:
                        if filing_dt.year != target_y:
                            continue

                # Quarter filtering:
                # 10-K is an Annual report covering the entire year; exclude if user asked for Q1/Q2/Q3
                if quarter:
                    q_clean = quarter.upper()
                    if form == "10-K":
                        if q_clean not in ("FY", "ANNUAL", "ALL"):
                            continue
                    elif form == "10-Q":
                        if fiscal_period.upper() != q_clean:
                            continue
                    elif form == "8-K":
                        if q_clean not in ("CURRENT", "ALL"):
                            continue

                accession_raw = filings_meta["accessionNumber"][i]
                accession = accession_raw.replace("-", "")
                primary_doc = filings_meta["primaryDocument"][i]
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
                safe_form = form.replace("/", "-")
                filename = f"{ticker.upper()}_{safe_form}_{fiscal_year}_{accession}.txt"

                extracted.append({
                    "ticker": ticker.upper(),
                    "cik": cik,
                    "form": form,
                    "filing_date": filing_date_str,
                    "report_date": report_date_str,
                    "year": fiscal_year,
                    "quarter": fiscal_period,
                    "calendar_quarter": calendar_quarter,
                    "fiscal_year_end": formatted_fye,
                    "accession": accession,
                    "primary_doc": primary_doc,
                    "url": doc_url,
                    "filename": filename,
                })
            return extracted

        results = _parse_filings_block(data.get("filings", {}).get("recent", {}))

        # Check paginated history if date range or specific year warrants it
        for file_info in data.get("filings", {}).get("files", []):
            p_from_dt = datetime.strptime(file_info["filingFrom"], "%Y-%m-%d").date()
            p_to_dt = datetime.strptime(file_info["filingTo"], "%Y-%m-%d").date()
            
            if start_dt and p_to_dt < start_dt:
                continue
            if end_dt and p_from_dt > end_dt:
                continue
            if year and (int(year) > p_to_dt.year + 1 or int(year) < p_from_dt.year - 1):
                continue

            p_url = f"https://data.sec.gov/submissions/{file_info['name']}"
            p_data = (await self._request(p_url)).json()
            results.extend(_parse_filings_block(p_data))

        # Sort most recent first
        results.sort(key=lambda x: x["filing_date"], reverse=True)
        return results

    async def download_and_save_filing_by_meta(
        self,
        filing_meta: Dict,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Downloads, parses, and cleans a specific filing identified by its discovery metadata.
        Saves cleanly to /Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{filename}.
        """
        target_dir = get_target_volume_dir(override_path=output_dir)
        filename = filing_meta.get("filename") or f"{filing_meta['ticker']}_{filing_meta['form']}_{filing_meta['year']}_{filing_meta['accession']}.txt"
        target_path = os.path.join(target_dir, filename)

        logger.info("Downloading %s for %s (%s) from %s...", filing_meta["form"], filing_meta["ticker"], filing_meta["year"], filing_meta["url"])
        resp = await self._request(filing_meta["url"])
        clean_text = clean_html_to_text(resp.text)

        header_text = (
            f"TICKER: {filing_meta['ticker']}\n"
            f"FORM_TYPE: {filing_meta['form']}\n"
            f"YEAR: {filing_meta['year']}\n"
            f"QUARTER: {filing_meta.get('quarter', 'N/A')}\n"
            f"FILING_DATE: {filing_meta['filing_date']}\n"
            f"PERIOD_END_DATE: {filing_meta['report_date']}\n"
            f"ACCESSION_NUMBER: {filing_meta['accession']}\n"
            f"SOURCE_URL: {filing_meta['url']}\n"
            f"{'='*80}\n\n"
        )
        full_content = header_text + clean_text

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(full_content)

        logger.info("Saved clean filing text to %s (%d chars)", target_path, len(full_content))
        return target_path

    async def list_filings(
        self,
        ticker: str,
        form_types: Sequence[str] = VALID_FORM_TYPES,
        year: Optional[int] = None,
    ) -> List[Dict]:
        """Backward-compatible discovery method."""
        return await self.discover_filings(ticker=ticker, form_types=form_types, year=year)

    async def download_and_save_filing(
        self,
        ticker: str,
        form: str,
        year: int,
        output_dir: Optional[str] = None,
    ) -> str:
        """Legacy helper downloading the top filing matching ticker, form, year."""
        filings = await self.discover_filings(ticker=ticker, form_types=[form], year=year)
        if not filings:
            raise FileNotFoundError(f"No SEC filing found for {ticker} ({form} {year}).")
        return await self.download_and_save_filing_by_meta(filings[0], output_dir=output_dir)


def _safe_run_coroutine(coro):
    """Executes a coroutine safely across both CLI scripts and active Databricks notebooks."""
    import concurrent.futures
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def discover_filings_sync(
    ticker: str,
    form_types: Optional[Sequence[str]] = None,
    year: Optional[int] = None,
    quarter: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    year_type: str = "fiscal",
) -> List[Dict]:
    """Synchronous discovery wrapper compatible with running notebook event loops."""
    async def _run():
        async with SECLoader() as loader:
            return await loader.discover_filings(
                ticker=ticker,
                form_types=form_types,
                year=year,
                quarter=quarter,
                start_date=start_date,
                end_date=end_date,
                year_type=year_type,
            )
    return _safe_run_coroutine(_run())


def load_specific_filing_sync(filing_meta: Dict, output_dir: Optional[str] = None) -> str:
    """Synchronous wrapper for downloading a specific discovered filing."""
    async def _run():
        async with SECLoader() as loader:
            return await loader.download_and_save_filing_by_meta(filing_meta, output_dir=output_dir)
    return _safe_run_coroutine(_run())


def load_sec_filing_sync(
    ticker: str,
    form: str,
    year: int,
    output_dir: Optional[str] = None,
) -> str:
    """Legacy synchronous convenience wrapper."""
    async def _run():
        async with SECLoader() as loader:
            return await loader.download_and_save_filing(ticker=ticker, form=form, year=year, output_dir=output_dir)
    return _safe_run_coroutine(_run())
