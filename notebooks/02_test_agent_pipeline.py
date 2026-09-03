# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 Databricks 2-Agent SEC Intelligence: E2E Testing, MLflow Tracing & App Deployment
# MAGIC > **Interactive Verification & Master Pipeline for SEC Intelligence on Databricks**
# MAGIC >
# MAGIC > - **Active Catalog**: `db_ai_strike_team`
# MAGIC > - **Active Schema**: `sec_intelligence`
# MAGIC > - **Serving Endpoint**: `databricks-meta-llama-3-3-70b-instruct`
# MAGIC > - **Vector Search Endpoint**: `sec_vs_endpoint`
# MAGIC >
# MAGIC > This notebook guides you through the complete interactive verification workflow:
# MAGIC > 1. **SEC Discovery & Selective Ingestion** (EDGAR -> UC Volume -> Delta Chunks + CDF)
# MAGIC > 2. **Delta Validation with SQL & Change Data Feed**
# MAGIC > 3. **2-Agent Intelligence Execution with MLflow 3.x Tracing**
# MAGIC > 4. **Delta-Backed HITL Memory Loop** (Critique persistence & prompt adaptation)
# MAGIC > 5. **Databricks Apps One-Click Deployment**
# MAGIC 
# MAGIC %pip install openai-agents>=0.2.0 databricks-vectorsearch>=0.40 databricks-mcp>=0.9.0 pyyaml>=6.0.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## ⚙️ Step 1: Environment & Path Initialization

# COMMAND ----------
import sys
import os
import yaml
import json

# Ensure repository root is in sys.path
repo_root = os.path.abspath(".")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Load configuration from app.yaml
if os.path.exists("app.yaml"):
    with open("app.yaml", "r") as f:
        app_config = yaml.safe_load(f)
        for env_var in app_config.get("env", []):
            os.environ[env_var["name"]] = str(env_var["value"])
    print("✅ Successfully loaded configurations from app.yaml")
else:
    os.environ["DATABRICKS_CATALOG"] = "db_ai_strike_team"
    os.environ["DATABRICKS_SCHEMA"] = "sec_intelligence"
    print("ℹ️ Using default environment parameters")

from config import (
    DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA,
    FULL_SCHEMA,
    CHUNKS_TABLE,
    FEEDBACK_TABLE,
    VS_INDEX_NAME,
    SERVING_ENDPOINT,
)

print(f"Target Namespace:       {FULL_SCHEMA}")
print(f"Chunks Delta Table:     {CHUNKS_TABLE}")
print(f"Feedback Delta Table:   {FEEDBACK_TABLE}")
print(f"Serving Model Endpoint: {SERVING_ENDPOINT}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🔎 Step 2: Test Granular SEC Discovery
# MAGIC Queries SEC EDGAR with compliant rate-limiting (10 req/s) to list available filings for a ticker.

# COMMAND ----------
from data_pipeline.sec_loader import discover_filings_sync
import pandas as pd

target_ticker = "NVDA"
target_form = "10-K"
target_year = 2024

print(f"Discovering {target_form} filings for {target_ticker} in {target_year}...")
filings = discover_filings_sync(
    ticker=target_ticker,
    form_types=[target_form],
    year=target_year,
)

print(f"Found {len(filings)} filings on SEC EDGAR:")
df_filings = pd.DataFrame(filings)
display(df_filings)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📥 Step 3: Trigger Background Ingestion & Vector Indexing
# MAGIC Ingests the filing: downloads from SEC, sanitizes HTML, writes clean text to UC Volume, extracts deterministic chunks, and idempotently upserts to Delta.

# COMMAND ----------
# MAGIC %sh
# MAGIC python jobs/ingest_sec_job.py --ticker NVDA --form 10-K --year 2024

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📊 Step 4: Verify Delta Table & Change Data Feed (CDF) in SQL
# MAGIC Check the ingested chunks and inspect the Delta Change Data Feed to confirm chunk idempotency and change tracking.

# COMMAND ----------
# MAGIC %sql
# MAGIC -- 1. Check indexed chunks count by ticker, form, year, and accession
# MAGIC SELECT 
# MAGIC     ticker, 
# MAGIC     form_type, 
# MAGIC     year, 
# MAGIC     accession, 
# MAGIC     count(*) AS total_chunks,
# MAGIC     min(length(content)) AS min_chunk_len,
# MAGIC     max(length(content)) AS max_chunk_len
# MAGIC FROM db_ai_strike_team.sec_intelligence.sec_filing_chunks
# MAGIC GROUP BY ALL;

# COMMAND ----------
# MAGIC %sql
# MAGIC -- 2. Inspect Change Data Feed (CDF) entries for real-time streaming to Vector Search
# MAGIC SELECT 
# MAGIC     _change_type, 
# MAGIC     _commit_version, 
# MAGIC     _commit_timestamp, 
# MAGIC     chunk_id, 
# MAGIC     ticker, 
# MAGIC     form_type, 
# MAGIC     year
# MAGIC FROM table_changes('db_ai_strike_team.sec_intelligence.sec_filing_chunks', 1)
# MAGIC LIMIT 10;

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🤖 Step 5: Test 2-Agent Engine with MLflow Tracing
# MAGIC Executes the full 2-agent intelligence loop:
# MAGIC 1. **Supervisor Agent**: Plans analysis and checks past feedback memory.
# MAGIC 2. **SEC Retrieval Agent (Subagent)**: Checks index status via MCP, decomposes query into technical angles, and retrieves evidence chunks.
# MAGIC 3. **Supervisor Synthesis**: Assembles executive research brief with exact chunk citations.

# COMMAND ----------
import mlflow
from agent.supervisor import SECSupervisorAgent

# Set or create an active MLflow experiment
mlflow.set_experiment(f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/sec_intelligence_experiment")

supervisor = SECSupervisorAgent()

test_query = "What drove NVIDIA's Data Center revenue growth and gross margin changes in fiscal 2024?"

print(f"Querying 2-Agent SEC Intelligence System:\n'{test_query}'\n")

response = supervisor.run_full_flow(
    user_query=test_query,
    ticker="NVDA",
    form_type="10-K",
    year=2024,
)

print("=" * 80)
print("📑 EXECUTIVE INVESTMENT BRIEF")
print("=" * 80)
print(response["report"])

# COMMAND ----------
# MAGIC %md
# MAGIC ### 🔍 View Live MLflow Traces
# MAGIC 1. In the **Right Sidebar** of this notebook, click on the **Experiments / Traces** icon.
# MAGIC 2. Click the latest trace to inspect the hierarchical flamegraph:
# MAGIC    - `supervisor_plan`: Intent formulation & feedback memory retrieval.
# MAGIC    - `retrieval_agent`: Vector Search similarity queries, chunk scores, and citations.
# MAGIC    - `final_synthesis`: LLM prompt context and Wall Street brief generation.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🧠 Step 6: Test Human-in-the-Loop (HITL) Feedback Memory Loop
# MAGIC Test recording user corrections and guidelines to Delta memory, and verify that the Supervisor automatically retrieves and adheres to them on subsequent runs.

# COMMAND ----------
from tools.uc_tools import record_feedback, get_relevant_feedback

# 1. Simulate a user providing constructive feedback and ground-truth correction
critique_status = record_feedback(
    query="gross margin drivers",
    ticker="NVDA",
    rating="NEGATIVE",
    feedback_text="Always break down Compute vs Networking gross margin drivers separately in the financial metrics section.",
    corrected_context="In fiscal 2024, Data Center compute revenue grew 217% to $39.5B while networking grew 133% to $8.0B.",
)
print("Feedback Record Status:", critique_status)

# 2. Retrieve past feedback memory for NVDA
memory_context = get_relevant_feedback(ticker="NVDA", query_topic="gross margin drivers")
print("\nRetrieved Memory Context for NVDA:")
print(memory_context)

# COMMAND ----------
# MAGIC %sql
# MAGIC -- Verify that feedback is persisted in the Delta memory table
# MAGIC SELECT 
# MAGIC     timestamp, 
# MAGIC     ticker, 
# MAGIC     rating, 
# MAGIC     feedback_text, 
# MAGIC     corrected_context
# MAGIC FROM db_ai_strike_team.sec_intelligence.agent_feedback
# MAGIC ORDER BY timestamp DESC 
# MAGIC LIMIT 5;

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🔁 Step 7: Verify Adaptive Re-Run with Memory Injected
# MAGIC Run the Supervisor again. Notice that the Supervisor now includes the user's critique from memory in its research brief!

# COMMAND ----------
re_run_response = supervisor.run_full_flow(
    user_query="Analyze NVIDIA's gross margin performance and segment mix for 2024.",
    ticker="NVDA",
    form_type="10-K",
    year=2024,
)

print(re_run_response["report"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🌐 Step 8: Publish as a Serverless Databricks App
# MAGIC Once verified in this notebook, deploy the interactive Streamlit UI to Databricks Apps.
# MAGIC 
# MAGIC Run the following commands from your terminal or Databricks CLI:
# MAGIC 
# MAGIC ```bash
# MAGIC # 1. Create the Databricks App definition (one-time)
# MAGIC databricks apps create sec-intelligence-agent
# MAGIC 
# MAGIC # 2. Deploy your repository source code to the Databricks App
# MAGIC databricks apps deploy sec-intelligence-agent --source-code-path /Workspace/Users/<your-username>/Investment_Agent_DB
# MAGIC ```
# MAGIC 
# MAGIC ### What happens upon deployment:
# MAGIC - Databricks builds a serverless container running `streamlit run app.py`.
# MAGIC - Reads configurations directly from `app.yaml`.
# MAGIC - Provides a secure, role-based URL hosted within your Databricks workspace.

