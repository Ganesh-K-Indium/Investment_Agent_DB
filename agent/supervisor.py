"""
Supervisor & Human-in-the-Loop (HITL) Agent (Agent 1) built with Databricks / OpenAI Agents SDK.
Features:
- Imports registered Unity Catalog tools over Databricks MCP endpoint.
- Connects sub-agent (SEC_Retrieval_Agent) via handoff and tool orchestration.
- Integrates memory retrieval from agent_feedback Delta table.
- Wraps reasoning phases in MLflow trace spans: supervisor_plan, retrieval_agent, final_synthesis.
"""

import os
import json
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional, List

from databricks.sdk import WorkspaceClient
from openai import OpenAI, AsyncOpenAI
from agents import Agent, Runner, handoff, set_default_openai_client

from config import (
    SERVING_ENDPOINT,
    DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA,
)
from tools.uc_tools import get_relevant_feedback
from tools.mcp_tools import get_databricks_mcp_server, get_databricks_mcp_agent_tools
from agent.retriever import SECRetrievalAgent

logger = logging.getLogger("agent.supervisor")


@contextmanager
def safe_mlflow_span(span_name: str):
    """Context manager executing inside an MLflow trace span with graceful fallback."""
    try:
        import mlflow
        if hasattr(mlflow, "start_span"):
            with mlflow.start_span(span_name) as span:
                yield span
        else:
            yield None
    except Exception as exc:
        logger.debug("MLflow span '%s' error or unavailable: %s", span_name, exc)
        yield None


def create_supervisor_agent(
    retrieval_agent_instance: Agent,
    model_serving_endpoint: str = SERVING_ENDPOINT,
) -> Agent:
    """
    Constructs the Main Supervisor Agent using Databricks MCP tools and Agents SDK.
    Directly passes the registered UC memory tools and registers the retrieval agent via handoffs.
    """
    instructions = (
        "You are the Lead Financial Research Supervisor and Principal Synthesizer on Databricks.\n"
        "Your mission is to formulate thorough investment briefs based on SEC filing disclosures and user feedback memory.\n"
        "Workflow:\n"
        "1. First, call `get_relevant_feedback` (imported via Databricks Unity Catalog MCP) to check past company critiques.\n"
        "2. Formulate a structured research plan incorporating past guidelines.\n"
        "3. Hand off or dispatch the search request to `SEC_Retrieval_Agent` to perform technical vector retrieval.\n"
        "4. Synthesize all findings and excerpts into an executive-ready brief with explicit chunk citations."
    )

    # 1. Fetch registered MCP tools from Databricks Unity Catalog
    mcp_tools = get_databricks_mcp_agent_tools(catalog=DATABRICKS_CATALOG, schema=DATABRICKS_SCHEMA)
    supervisor_tools = [t for t in mcp_tools if t.name in ("get_relevant_feedback", "record_feedback")]

    # 2. Connect managed Databricks MCP server if available
    mcp_server = get_databricks_mcp_server(catalog=DATABRICKS_CATALOG, schema=DATABRICKS_SCHEMA)
    mcp_servers = [mcp_server] if mcp_server else []

    return Agent(
        name="SEC_Supervisor_Agent",
        instructions=instructions,
        model=model_serving_endpoint,
        tools=supervisor_tools,
        handoffs=[retrieval_agent_instance],
        mcp_servers=mcp_servers,
    )


class SECSupervisorAgent:
    """Supervisor and HITL orchestrator agent."""

    def __init__(self, model_serving_endpoint: str = SERVING_ENDPOINT):
        self.endpoint_name = model_serving_endpoint
        from config import get_databricks_host_and_token, get_workspace_client
        self.workspace_client = get_workspace_client()
        host, token = get_databricks_host_and_token()

        self.llm_client = OpenAI(
            api_key=token,
            base_url=f"{host}/serving-endpoints",
        )

        try:
            async_client = AsyncOpenAI(api_key=token, base_url=f"{host}/serving-endpoints")
            set_default_openai_client(async_client, use_for_tracing=False)
        except Exception as exc:
            logger.debug("set_default_openai_client deferred: %s", exc)

        # Initialize Retrieval Agent (Agent 2)
        self.retriever = SECRetrievalAgent(model_serving_endpoint=model_serving_endpoint)

        # Build Supervisor Agent (Agent 1) with MCP tools and retrieval sub-agent handoff
        self.agent = create_supervisor_agent(
            retrieval_agent_instance=self.retriever.agent,
            model_serving_endpoint=model_serving_endpoint,
        )

    def plan_task(
        self,
        user_query: str,
        ticker: str,
        form_type: str = "10-K",
        year: int = 2023,
    ) -> Dict[str, Any]:
        """
        Step 1: Queries past Delta feedback memory and designs an execution plan.
        Wrapped in MLflow span: 'supervisor_plan'.
        """
        with safe_mlflow_span("supervisor_plan") as span:
            clean_ticker = ticker.strip().upper()
            logger.info("Retrieving past feedback memory for %s...", clean_ticker)
            past_feedback = get_relevant_feedback(ticker=clean_ticker, query_topic=user_query)

            system_prompt = (
                "You are a Lead Financial Research Director overseeing an SEC filings intelligence team. "
                "Formulate a structured research plan to answer the user's question.\n"
                "Incorporate any past user guidelines, corrections, or critiques provided.\n"
                "Output ONLY a valid JSON object with keys: target_ticker, form_type, fiscal_year, planned_sub_queries, analytical_focus."
            )

            user_prompt = (
                f"User Question: {user_query}\n"
                f"Company Ticker: {clean_ticker}\n"
                f"Form Type: {form_type}\n"
                f"Year: {year}\n\n"
                f"Memory (Past Critiques):\n{past_feedback}\n\n"
                "Generate the structured JSON research plan:"
            )

            try:
                response = self.llm_client.chat.completions.create(
                    model=self.endpoint_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=350,
                )
                raw_text = response.choices[0].message.content.strip()
                if raw_text.startswith("```"):
                    lines = raw_text.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    raw_text = "\n".join(lines).strip()
                plan_data = json.loads(raw_text)
            except Exception as e:
                logger.warning("LLM planning fallback triggered: %s", e)
                plan_data = {
                    "target_ticker": clean_ticker,
                    "form_type": form_type,
                    "fiscal_year": year,
                    "planned_sub_queries": [
                        f"{clean_ticker} {form_type} {user_query}",
                        f"{clean_ticker} management discussion and analysis {user_query}",
                    ],
                    "analytical_focus": f"Analyze SEC disclosures for {clean_ticker} addressing: {user_query}",
                }

            plan_data["past_feedback"] = past_feedback
            plan_data["user_query"] = user_query
            if span and hasattr(span, "set_inputs"):
                try:
                    span.set_inputs({"query": user_query, "ticker": clean_ticker})
                    span.set_outputs(plan_data)
                except Exception:
                    pass

            return plan_data

    def execute_retrieval(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step 2: Dispatches retrieval request to Agent 2 (retriever).
        Wrapped in MLflow span: 'retrieval_agent'.
        """
        with safe_mlflow_span("retrieval_agent") as span:
            ticker = plan.get("target_ticker", "UNKNOWN")
            form_type = plan.get("form_type", "10-K")
            year = plan.get("fiscal_year", 2023)
            user_query = plan.get("user_query", "")

            retrieval_result = self.retriever.retrieve_and_format(
                user_query=user_query,
                ticker=ticker,
                form_type=form_type,
                year=year,
            )

            if span and hasattr(span, "set_outputs"):
                try:
                    span.set_outputs({
                        "status": retrieval_result.get("status"),
                        "chunks_retrieved": len(retrieval_result.get("evidence_chunks", [])),
                    })
                except Exception:
                    pass

            return retrieval_result

    def synthesize_report(
        self,
        user_query: str,
        plan: Dict[str, Any],
        retrieval_result: Dict[str, Any],
    ) -> str:
        """
        Step 3: Synthesizes evidence and past feedback into a comprehensive research brief.
        Wrapped in MLflow span: 'final_synthesis'.
        """
        with safe_mlflow_span("final_synthesis") as span:
            if not retrieval_result.get("success") and retrieval_result.get("alert"):
                return retrieval_result["alert"]

            evidence_text = retrieval_result.get("formatted_evidence", "No evidence chunks available.")
            past_feedback = plan.get("past_feedback", "")
            ticker = plan.get("target_ticker", "UNKNOWN")
            form_type = plan.get("form_type", "10-K")
            year = plan.get("fiscal_year", 2023)

            system_prompt = (
                "You are a Senior Wall Street Equity Research Analyst and Principal AI Synthesizer on Databricks. "
                "Synthesize a rigorous, executive-ready investment research brief based ONLY on the provided SEC filing evidence "
                "and adhering strictly to the user's past feedback guidelines.\n\n"
                "Format Requirements:\n"
                "1. **Executive Summary**: Core takeaway and direct answer.\n"
                "2. **Detailed Financial Findings & Metrics**: Breakdowns, percentages, segment data, and operational drivers.\n"
                "3. **Risk & Management Outlook**: Notes from MD&A or disclosures.\n"
                "4. **Citations & Sources**: Explicitly reference the retrieved chunks (e.g. [TICKER FORM YEAR | Chunk N]).\n"
                "5. Follow any past user feedback or formatting instructions explicitly."
            )

            user_prompt = (
                f"# RESEARCH BRIEF REQUEST\n"
                f"**Company**: {ticker} | **Filing**: {form_type} {year}\n"
                f"**Question**: {user_query}\n\n"
                f"## USER FEEDBACK MEMORY:\n{past_feedback}\n\n"
                f"## RETRIEVED SEC EVIDENCE:\n{evidence_text}\n\n"
                f"Generate the comprehensive research brief now:"
            )

            try:
                response = self.llm_client.chat.completions.create(
                    model=self.endpoint_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=1500,
                )
                final_brief = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error("Synthesis LLM call failed: %s", e)
                final_brief = (
                    f"### Research Brief for {ticker} ({form_type} {year})\n\n"
                    f"**Direct Query**: {user_query}\n\n"
                    f"#### Retrieved Evidence Overview:\n{evidence_text}\n\n"
                    f"*(Note: Model serving endpoint returned error: {e})*"
                )

            if span and hasattr(span, "set_outputs"):
                try:
                    span.set_outputs({"brief_length": len(final_brief)})
                except Exception:
                    pass

            return final_brief

    def run_full_flow(
        self,
        user_query: str,
        ticker: str,
        form_type: str = "10-K",
        year: int = 2023,
        approved_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Executes the end-to-end multi-agent flow."""
        plan = approved_plan or self.plan_task(
            user_query=user_query,
            ticker=ticker,
            form_type=form_type,
            year=year,
        )

        retrieval_result = self.execute_retrieval(plan)
        report = self.synthesize_report(user_query, plan, retrieval_result)

        return {
            "plan": plan,
            "retrieval": retrieval_result,
            "report": report,
        }
