"""
Databricks Unity Catalog MCP Tool Loader.
Connects to Databricks managed MCP endpoints (e.g. /api/2.0/mcp/functions/{catalog}/{schema})
using DatabricksMCPClient to dynamically discover and import registered Unity Catalog tools.
Transforms MCP tools into Agent tools compatible with the OpenAI Agents SDK (agents.Agent).
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from agents import FunctionTool
from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams

from config import DATABRICKS_CATALOG, DATABRICKS_SCHEMA
from tools.uc_tools import check_filing_status, record_feedback, get_relevant_feedback

logger = logging.getLogger("tools.mcp_tools")


def get_mcp_server_url(catalog: str = DATABRICKS_CATALOG, schema: str = DATABRICKS_SCHEMA) -> str:
    """Returns the Databricks managed MCP URL for Unity Catalog functions."""
    try:
        w = WorkspaceClient()
        host = (w.config.host or "https://databricks.local").rstrip("/")
    except Exception:
        host = os.getenv("DATABRICKS_HOST", "https://databricks.local").rstrip("/")

    return f"{host}/api/2.0/mcp/functions/{catalog}/{schema}"


def get_databricks_mcp_server(
    catalog: str = DATABRICKS_CATALOG,
    schema: str = DATABRICKS_SCHEMA,
) -> Optional[MCPServerStreamableHttp]:
    """
    Constructs an MCPServerStreamableHttp instance targeting the Databricks UC functions MCP URL.
    Can be passed directly to Agent(..., mcp_servers=[mcp_server]).
    """
    try:
        w = WorkspaceClient()
        host = (w.config.host or "").rstrip("/")
        token = w.config.token or ""
        if not host or not token:
            return None

        mcp_url = f"{host}/api/2.0/mcp/functions/{catalog}/{schema}"
        params = MCPServerStreamableHttpParams(
            url=mcp_url,
            headers={"Authorization": f"Bearer {token}"},
        )
        return MCPServerStreamableHttp(params=params, name=f"uc_mcp_{catalog}_{schema}")
    except Exception as exc:
        logger.debug("MCP server creation: %s", exc)
        return None


def get_databricks_mcp_agent_tools(
    catalog: str = DATABRICKS_CATALOG,
    schema: str = DATABRICKS_SCHEMA,
) -> List[FunctionTool]:
    """
    Connects to Databricks MCP server via DatabricksMCPClient,
    discovers registered UC functions over MCP, and converts them into Agent tools.
    Falls back gracefully if workspace is offline or unauthenticated.
    """
    try:
        w = WorkspaceClient()
        host = (w.config.host or "").rstrip("/")
        if not host:
            raise ValueError("No Databricks workspace host available")

        mcp_url = f"{host}/api/2.0/mcp/functions/{catalog}/{schema}"
        logger.info("Connecting to Databricks MCP server at %s...", mcp_url)

        mcp_client = DatabricksMCPClient(server_url=mcp_url, workspace_client=w)
        mcp_tools = mcp_client.list_tools()
        logger.info("Discovered %d MCP tools from %s", len(mcp_tools), mcp_url)

        agent_tools = []
        for mt in mcp_tools:
            raw_name = mt.name
            clean_name = raw_name.split(".")[-1]
            desc = mt.description or f"Unity Catalog tool {clean_name}"
            input_schema = getattr(mt, "inputSchema", None) or {"type": "object", "properties": {}}

            # Closure for async tool invocation over MCP
            def _make_invoker(tool_name: str):
                async def _invoke(ctx, input_json: str):
                    args = json.loads(input_json) if input_json else {}
                    res = await mcp_client.acall_tool(tool_name, args)
                    texts = [c.text for c in res.content if hasattr(c, "text")]
                    return "\n".join(texts) if texts else str(res)
                return _invoke

            ft = FunctionTool(
                name=clean_name,
                description=desc,
                params_json_schema=input_schema if isinstance(input_schema, dict) else {},
                on_invoke_tool=_make_invoker(raw_name),
            )
            agent_tools.append(ft)

        if agent_tools:
            return agent_tools
    except Exception as exc:
        logger.warning("Databricks MCP tool discovery deferred/offline: %s. Using direct fallback tools.", exc)

    # Graceful fallback: Wrap local functions as FunctionTool
    fallback_tools = []

    async def _invoke_status(ctx, input_json: str):
        args = json.loads(input_json) if input_json else {}
        return check_filing_status(
            ticker=args.get("ticker", ""),
            form_type=args.get("form_type", "10-K"),
            year=int(args.get("year", 2024)),
        )

    fallback_tools.append(FunctionTool(
        name="check_filing_status",
        description="Checks if filing chunks exist in Unity Catalog sec_filing_chunks.",
        params_json_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker"},
                "form_type": {"type": "string", "description": "10-K, 10-Q, 8-K"},
                "year": {"type": "integer", "description": "Year"},
            },
            "required": ["ticker", "form_type", "year"],
        },
        on_invoke_tool=_invoke_status,
    ))

    async def _invoke_feedback_get(ctx, input_json: str):
        args = json.loads(input_json) if input_json else {}
        return get_relevant_feedback(
            ticker=args.get("ticker", ""),
            query_topic=args.get("query_topic", ""),
        )

    fallback_tools.append(FunctionTool(
        name="get_relevant_feedback",
        description="Queries agent_feedback Delta table for past user critiques and guidelines.",
        params_json_schema={
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker"},
                "query_topic": {"type": "string", "description": "Topic or question"},
            },
            "required": ["ticker", "query_topic"],
        },
        on_invoke_tool=_invoke_feedback_get,
    ))

    async def _invoke_feedback_record(ctx, input_json: str):
        args = json.loads(input_json) if input_json else {}
        return record_feedback(
            query=args.get("query", ""),
            ticker=args.get("ticker", ""),
            rating=args.get("rating", ""),
            feedback_text=args.get("feedback_text", ""),
            corrected_context=args.get("corrected_context", ""),
        )

    fallback_tools.append(FunctionTool(
        name="record_feedback",
        description="Records user corrections and feedback into Unity Catalog Delta memory.",
        params_json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "ticker": {"type": "string"},
                "rating": {"type": "string"},
                "feedback_text": {"type": "string"},
                "corrected_context": {"type": "string"},
            },
            "required": ["query", "ticker", "rating", "feedback_text"],
        },
        on_invoke_tool=_invoke_feedback_record,
    ))

    return fallback_tools

