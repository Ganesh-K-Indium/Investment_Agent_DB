"""
Intelligent SEC Retrieval Agent (Agent 2).
Responsible for:
1. Pre-check: Invokes check_filing_status to ensure data readiness.
2. Query Decomposition: Breaks complex user financial questions into 2-3 technical search angles.
3. Vector Search Execution: Queries Databricks Vector Search index with metadata filters.
4. Deduplication & Evidence Formatting: Deduplicates chunks by chunk_id and formats structured citations.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from databricks.sdk import WorkspaceClient
from openai import OpenAI

from config import (
    SERVING_ENDPOINT,
    VECTOR_SEARCH_ENDPOINT,
    VS_INDEX_NAME,
)
from tools.uc_tools import check_filing_status

logger = logging.getLogger("agent.retriever")


class SECRetrievalAgent:
    """Intelligent SEC retrieval agent executing targeted multi-perspective searches."""

    def __init__(self, model_serving_endpoint: str = SERVING_ENDPOINT):
        self.endpoint_name = model_serving_endpoint
        try:
            self.workspace_client = WorkspaceClient()
            host = (self.workspace_client.config.host or "https://databricks.local").rstrip("/")
            token = self.workspace_client.config.token or "no-token"
        except Exception as e:
            logger.warning("Databricks WorkspaceClient auth unavailable at init: %s", e)
            self.workspace_client = None
            host = os.getenv("DATABRICKS_HOST", "https://databricks.local").rstrip("/")
            token = os.getenv("DATABRICKS_TOKEN", "no-token")

        self.llm_client = OpenAI(
            api_key=token,
            base_url=f"{host}/serving-endpoints",
        )

    def verify_filing_availability(self, ticker: str, form_type: str, year: int) -> Dict[str, Any]:
        """Pre-check step: verifies if the target filing is already indexed in Unity Catalog."""
        raw_status = check_filing_status(ticker=ticker, form_type=form_type, year=year)
        try:
            return json.loads(raw_status)
        except Exception:
            return {"status": "UNKNOWN", "chunk_count": 0, "raw": raw_status}

    def decompose_query(self, user_query: str, ticker: str, form_type: str, year: int) -> List[str]:
        """
        Decomposes an arbitrary financial question into 2-3 specific technical SEC search queries.
        E.g. A question on gross margins decomposes into:
        - Segment revenue and cost of goods sold breakdown
        - Management Discussion and Analysis (MD&A) margin commentary
        - Gross profit disclosures and accounting policies
        """
        system_prompt = (
            "You are an expert Wall Street research assistant and SEC filing specialist. "
            "Given a user's analytical question regarding an SEC filing, decompose the question into "
            "2 to 3 targeted, highly technical sub-queries optimized for semantic vector search over 10-K/10-Q/8-K texts.\n"
            "Focus on specific financial disclosure terms (e.g. 'Segment Revenue', 'Cost of Sales', 'MD&A', 'Risk Factors', 'Liquidity and Capital Resources').\n"
            "Output ONLY a valid JSON list of strings, with no surrounding formatting or preamble.\n"
            "Example: [\"Segment revenue and gross profit breakdown\", \"MD&A gross margin variance analysis\"]"
        )
        user_prompt = (
            f"Ticker: {ticker.upper()}\n"
            f"Form Type: {form_type}\n"
            f"Year: {year}\n"
            f"Question: {user_query}\n\n"
            "Decompose into 2-3 targeted vector search sub-queries:"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.endpoint_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            content = response.choices[0].message.content.strip()
            # Clean possible markdown code fences
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            sub_queries = json.loads(content)
            if isinstance(sub_queries, list) and len(sub_queries) > 0:
                return [str(q).strip() for q in sub_queries[:3]]
        except Exception as e:
            logger.warning("LLM query decomposition fallback triggered: %s", e)

        # Fallback sub-queries if LLM call fails or returns non-JSON
        return [
            f"{ticker} {form_type} {user_query}",
            f"{ticker} MD&A financial performance and results of operations {user_query}",
            f"{ticker} consolidated financial statements notes {user_query}",
        ]

    def execute_vector_search(
        self,
        sub_queries: List[str],
        ticker: str,
        form_type: Optional[str] = None,
        year: Optional[int] = None,
        k: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Queries Databricks Vector Search for each sub-query, applies metadata filters,
        and deduplicates chunks by chunk_id.
        """
        from databricks.vector_search.client import VectorSearchClient

        vsc = VectorSearchClient()
        try:
            index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=VS_INDEX_NAME)
        except Exception as exc:
            logger.error("Vector search index '%s' unavailable on endpoint '%s': %s", VS_INDEX_NAME, VECTOR_SEARCH_ENDPOINT, exc)
            return []

        filters = {"ticker": ticker.upper()}
        if form_type:
            filters["form_type"] = form_type.upper()
        if year:
            filters["year"] = int(year)

        dedup_chunks = {}
        for query_text in sub_queries:
            try:
                search_resp = index.similarity_search(
                    query_text=query_text,
                    columns=["chunk_id", "ticker", "form_type", "year", "content"],
                    num_results=k,
                    filters=filters,
                )
                
                # Databricks Vector Search returns dict with 'manifest' and 'result.data_array'
                result_data = search_resp.get("result", {}).get("data_array", [])
                columns = [c.get("name") for c in search_resp.get("manifest", {}).get("columns", [])]

                for row in result_data:
                    row_dict = dict(zip(columns, row)) if columns else {}
                    chunk_id = row_dict.get("chunk_id", str(row[0]) if row else "")
                    if chunk_id and chunk_id not in dedup_chunks:
                        dedup_chunks[chunk_id] = {
                            "chunk_id": chunk_id,
                            "ticker": row_dict.get("ticker", ticker.upper()),
                            "form_type": row_dict.get("form_type", form_type),
                            "year": row_dict.get("year", year),
                            "content": row_dict.get("content", str(row[-1]) if row else ""),
                            "matched_query": query_text,
                        }
            except Exception as e:
                logger.warning("Vector search query '%s' failed: %s", query_text, e)

        return list(dedup_chunks.values())

    def retrieve_and_format(
        self,
        user_query: str,
        ticker: str,
        form_type: str = "10-K",
        year: int = 2023,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end retrieval:
        1. Checks filing status. Returns actionable alert if not indexed.
        2. Decomposes query into 2-3 technical sub-queries.
        3. Executes similarity searches and deduplicates chunks.
        4. Formats citations and contextual evidence.
        """
        # Step 1: Pre-check status
        status_info = self.verify_filing_availability(ticker=ticker, form_type=form_type, year=year)
        if status_info.get("status") != "INDEXED" or status_info.get("chunk_count", 0) == 0:
            alert_msg = (
                f"⚠️ **ALERT: SEC Filing Not Indexed**\n\n"
                f"The target filing for **{ticker.upper()} ({form_type} {year})** has not been ingested yet "
                f"in `{VS_INDEX_NAME}` (found 0 chunks).\n\n"
                f"**Action Required**: Please run the background ingestion job via the sidebar or CLI (`jobs/ingest_sec_job.py --ticker {ticker.upper()} --form {form_type} --year {year}`) "
                f"before running this query."
            )
            return {
                "success": False,
                "status": "NOT_FOUND",
                "alert": alert_msg,
                "sub_queries": [],
                "evidence_chunks": [],
            }

        # Step 2: Decompose query
        sub_queries = self.decompose_query(
            user_query=user_query,
            ticker=ticker,
            form_type=form_type,
            year=year,
        )

        # Step 3: Vector search with deduplication
        chunks = self.execute_vector_search(
            sub_queries=sub_queries,
            ticker=ticker,
            form_type=form_type,
            year=year,
        )

        # Step 4: Format evidence and citations
        formatted_evidence = []
        for idx, chunk in enumerate(chunks, 1):
            citation_label = f"[{ticker.upper()} {chunk['form_type']} {chunk['year']} | Chunk {idx}]"
            formatted_evidence.append(
                f"### Citation {citation_label}\n"
                f"**Query Anchor**: *{chunk['matched_query']}*\n"
                f"**Excerpt**:\n> {chunk['content']}\n"
            )

        return {
            "success": True,
            "status": "INDEXED",
            "alert": None,
            "sub_queries": sub_queries,
            "evidence_chunks": chunks,
            "formatted_evidence": "\n\n".join(formatted_evidence),
        }
