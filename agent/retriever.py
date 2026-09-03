"""
Intelligent SEC Retrieval Agent (Agent 2) built with Databricks / OpenAI Agents SDK.
Features:
- Imports registered Unity Catalog tools over Databricks MCP endpoint.
- Uses DatabricksMCPClient to dynamically list and execute MCP tools.
- Connected via Agent(name=..., instructions=..., model=..., mcp_servers=[...], tools=[...]).
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from databricks.sdk import WorkspaceClient
from openai import OpenAI, AsyncOpenAI
from agents import Agent, FunctionTool, set_default_openai_client, Runner

from config import (
    SERVING_ENDPOINT,
    VECTOR_SEARCH_ENDPOINT,
    VS_INDEX_NAME,
    DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA,
)
from tools.uc_tools import check_filing_status
from tools.mcp_tools import get_databricks_mcp_server, get_databricks_mcp_agent_tools

logger = logging.getLogger("agent.retriever")


def create_retrieval_agent(model_serving_endpoint: str = SERVING_ENDPOINT) -> Agent:
    """
    Constructs the SEC Retrieval Agent using Databricks MCP server tools and Agents SDK.
    """
    instructions = (
        "You are the Intelligent SEC Retrieval Agent specialized in financial filings research.\n"
        "Your mission is to find ground-truth financial disclosures from 10-K, 10-Q, and 8-K filings.\n"
        "Workflow:\n"
        "1. First, invoke `check_filing_status` (imported via Databricks Unity Catalog MCP) to verify the filing is ready.\n"
        "   If status is NOT_FOUND or chunk_count is 0, report immediately that background ingestion is needed.\n"
        "2. Decompose user inquiries into 2 to 3 targeted technical sub-queries (e.g. Segment Revenue, MD&A, Cost of Sales).\n"
        "3. Search vector index to retrieve evidence chunks with metadata filters.\n"
        "4. Deduplicate and return verified excerpts with chunk citations."
    )

    # 1. Fetch registered MCP tools from Databricks Unity Catalog
    mcp_tools = get_databricks_mcp_agent_tools(catalog=DATABRICKS_CATALOG, schema=DATABRICKS_SCHEMA)
    # Filter tools relevant to retrieval
    retrieval_tools = [t for t in mcp_tools if t.name in ("check_filing_status", "check_accession_status")]

    # 2. Add vector search tool
    async def _invoke_vs(ctx, input_json: str):
        args = json.loads(input_json) if input_json else {}
        q = args.get("query_text", "")
        tkr = args.get("ticker", "")
        frm = args.get("form_type", "10-K")
        yr = int(args.get("year", 2024))

        from databricks.vector_search.client import VectorSearchClient
        try:
            vsc = VectorSearchClient()
            idx = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=VS_INDEX_NAME)
            res = idx.similarity_search(
                query_text=q,
                columns=["chunk_id", "ticker", "form_type", "year", "content"],
                num_results=4,
                filters={"ticker": tkr.upper(), "form_type": frm.upper(), "year": yr},
            )
            rows = res.get("result", {}).get("data_array", [])
            lines = [f"[{r[0]}]: {r[-1]}" for r in rows if len(r) > 0]
            return "\n---\n".join(lines) if lines else f"No matches for '{q}'"
        except Exception as exc:
            return f"Vector search error: {exc}"

    retrieval_tools.append(FunctionTool(
        name="search_sec_filings",
        description="Executes vector search on Databricks Vector Search index with metadata filters.",
        params_json_schema={
            "type": "object",
            "properties": {
                "query_text": {"type": "string", "description": "Search phrase"},
                "ticker": {"type": "string", "description": "Stock ticker"},
                "form_type": {"type": "string", "description": "10-K, 10-Q, 8-K"},
                "year": {"type": "integer", "description": "Year"},
            },
            "required": ["query_text", "ticker"],
        },
        on_invoke_tool=_invoke_vs,
    ))

    # 3. Connect managed Databricks MCP server if available
    mcp_server = get_databricks_mcp_server(catalog=DATABRICKS_CATALOG, schema=DATABRICKS_SCHEMA)
    mcp_servers = [mcp_server] if mcp_server else []

    return Agent(
        name="SEC_Retrieval_Agent",
        instructions=instructions,
        model=model_serving_endpoint,
        tools=retrieval_tools,
        mcp_servers=mcp_servers,
    )


class SECRetrievalAgent:
    """SECRetrievalAgent orchestrating retrieval and wrapping Agent."""

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

        try:
            async_client = AsyncOpenAI(api_key=token, base_url=f"{host}/serving-endpoints")
            set_default_openai_client(async_client, use_for_tracing=False)
        except Exception as exc:
            logger.debug("set_default_openai_client deferred: %s", exc)

        # Build underlying Agent with MCP tools
        self.agent = create_retrieval_agent(model_serving_endpoint=model_serving_endpoint)

    def verify_filing_availability(self, ticker: str, form_type: str, year: int) -> Dict[str, Any]:
        """Pre-check step: verifies if target filing is already indexed."""
        raw_status = check_filing_status(ticker=ticker, form_type=form_type, year=year)
        try:
            return json.loads(raw_status)
        except Exception:
            return {"status": "UNKNOWN", "chunk_count": 0, "raw": raw_status}

    def decompose_query(self, user_query: str, ticker: str, form_type: str, year: int) -> List[str]:
        """Decomposes user question into 2-3 technical SEC search queries."""
        system_prompt = (
            "You are an expert Wall Street research assistant. "
            "Decompose the user's SEC filing question into 2 to 3 targeted technical sub-queries "
            "optimized for semantic search over financial disclosures. Output ONLY a valid JSON list of strings."
        )
        user_prompt = (
            f"Ticker: {ticker.upper()}\n"
            f"Form Type: {form_type}\n"
            f"Year: {year}\n"
            f"Question: {user_query}\n\n"
            "Decompose into 2-3 targeted sub-queries:"
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
            logger.warning("Query decomposition fallback: %s", e)

        return [
            f"{ticker} {form_type} {user_query}",
            f"{ticker} MD&A results of operations {user_query}",
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
        """Queries Vector Search index and deduplicates chunks by chunk_id."""
        from databricks.vector_search.client import VectorSearchClient

        try:
            vsc = VectorSearchClient()
            index = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=VS_INDEX_NAME)
        except Exception as exc:
            logger.error("Vector search index unavailable: %s", exc)
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
        """Executes full pre-check, multi-query retrieval, deduplication, and citation building."""
        status_info = self.verify_filing_availability(ticker=ticker, form_type=form_type, year=year)
        if status_info.get("status") != "INDEXED" or status_info.get("chunk_count", 0) == 0:
            alert_msg = (
                f"⚠️ **ALERT: SEC Filing Not Indexed**\n\n"
                f"The target filing for **{ticker.upper()} ({form_type} {year})** has not been ingested yet "
                f"in `{VS_INDEX_NAME}` (found 0 chunks).\n\n"
                f"**Action Required**: Please run the background ingestion job via the sidebar or CLI "
                f"(`jobs/ingest_sec_job.py --ticker {ticker.upper()} --form {form_type} --year {year}`) "
                f"before running this query."
            )
            return {
                "success": False,
                "status": "NOT_FOUND",
                "alert": alert_msg,
                "sub_queries": [],
                "evidence_chunks": [],
            }

        sub_queries = self.decompose_query(user_query=user_query, ticker=ticker, form_type=form_type, year=year)
        chunks = self.execute_vector_search(sub_queries=sub_queries, ticker=ticker, form_type=form_type, year=year)

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
