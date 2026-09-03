# Databricks notebook source
# MAGIC %md
# MAGIC # 🏛️ Databricks 2-Agent SEC Intelligence: Infrastructure & UC Governance Setup
# MAGIC > **Enterprise Agentic AI Production Pipeline on Databricks Data Intelligence Platform**
# MAGIC >
# MAGIC > - **Catalog**: `db_ai_strike_team`
# MAGIC > - **Schema**: `sec_intelligence`
# MAGIC > - **Volume**: `raw_filings`
# MAGIC > - **Vector Search Endpoint**: `sec_vs_endpoint`
# MAGIC > - **Delta Tables**: `sec_filing_chunks` (CDF enabled) & `agent_feedback` (HITL Memory)
# MAGIC > - **UC Governed Python Tools**: `check_filing_status`, `record_feedback`, `get_relevant_feedback`

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📦 Step 1: Install Dependencies & Restart Python Interpreter

# COMMAND ----------
# MAGIC %pip install databricks-sdk databricks-vectorsearch databricks-mcp openai-agents mlflow openai httpx beautifulsoup4 streamlit unitycatalog-ai pydantic pyyaml
# MAGIC dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## ⚙️ Step 2: Auto-Load Environment Configuration from `app.yaml`

# COMMAND ----------
import os
import sys
import yaml

# Set repo root in path
repo_root = os.path.abspath(".")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Load environment configuration from app.yaml
if os.path.exists("app.yaml"):
    with open("app.yaml", "r") as f:
        app_config = yaml.safe_load(f)
        for env_var in app_config.get("env", []):
            os.environ[env_var["name"]] = str(env_var["value"])
    print("✅ Loaded environment from app.yaml:")
else:
    os.environ["DATABRICKS_CATALOG"] = "db_ai_strike_team"
    os.environ["DATABRICKS_SCHEMA"] = "sec_intelligence"
    os.environ["DATABRICKS_VOLUME"] = "raw_filings"
    os.environ["VECTOR_SEARCH_ENDPOINT"] = "sec_vs_endpoint"
    print("ℹ️ Using default environment values:")

print(f" - CATALOG: {os.getenv('DATABRICKS_CATALOG')}")
print(f" - SCHEMA:  {os.getenv('DATABRICKS_SCHEMA')}")
print(f" - VOLUME:  {os.getenv('DATABRICKS_VOLUME')}")
print(f" - VS ENDPOINT: {os.getenv('VECTOR_SEARCH_ENDPOINT')}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🏛️ Step 3: Provision Unity Catalog Namespace, Volume & Delta Tables

# COMMAND ----------
# MAGIC %sql
# MAGIC -- 1. Switch to active team catalog and provision schema
# MAGIC USE CATALOG db_ai_strike_team;
# MAGIC CREATE SCHEMA IF NOT EXISTS sec_intelligence;
# MAGIC USE SCHEMA sec_intelligence;
# MAGIC 
# MAGIC -- 2. Create raw files volume for SEC filings
# MAGIC CREATE VOLUME IF NOT EXISTS raw_filings;
# MAGIC 
# MAGIC -- 3. Create Human-In-The-Loop feedback memory table
# MAGIC CREATE TABLE IF NOT EXISTS agent_feedback (
# MAGIC     feedback_id STRING NOT NULL,
# MAGIC     timestamp TIMESTAMP,
# MAGIC     ticker STRING,
# MAGIC     query STRING,
# MAGIC     rating STRING,
# MAGIC     feedback_text STRING,
# MAGIC     corrected_context STRING
# MAGIC );
# MAGIC 
# MAGIC -- 4. Create SEC filing chunks table with Change Data Feed (CDF) enabled
# MAGIC CREATE TABLE IF NOT EXISTS sec_filing_chunks (
# MAGIC     chunk_id STRING NOT NULL,
# MAGIC     ticker STRING,
# MAGIC     form_type STRING,
# MAGIC     year INT,
# MAGIC     content STRING,
# MAGIC     CONSTRAINT sec_chunks_pk PRIMARY KEY (chunk_id)
# MAGIC )
# MAGIC TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC 
# MAGIC -- 5. Verify active context
# MAGIC SELECT 
# MAGIC     current_catalog() AS active_catalog, 
# MAGIC     current_schema() AS active_schema, 
# MAGIC     current_user() AS active_user;

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🛠️ Step 4: Register Governed Unity Catalog Python Functions
# MAGIC Registers Python UDFs using `DatabricksFunctionClient.create_python_function` so they can be exposed via Databricks Managed MCP.

# COMMAND ----------
from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from tools.uc_tools import check_filing_status, record_feedback, get_relevant_feedback
from config import DATABRICKS_CATALOG, DATABRICKS_SCHEMA

print(f"Registering UC Python functions to {DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}...")

client = DatabricksFunctionClient()

# Bind active notebook Spark session to avoid detached sessions
client.spark = spark
client.spark.sql(f"USE CATALOG `{DATABRICKS_CATALOG}`")
client.spark.sql(f"USE SCHEMA `{DATABRICKS_SCHEMA}`")

tools_to_register = [
    check_filing_status,
    record_feedback,
    get_relevant_feedback,
]

for tool_func in tools_to_register:
    func_name = tool_func.__name__
    print(f" -> Registering Python function '{func_name}'...")
    client.create_python_function(
        func=tool_func,
        catalog=DATABRICKS_CATALOG,
        schema=DATABRICKS_SCHEMA,
        replace=True,
    )
    print(f"    ✅ Successfully registered: {DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.{func_name}")

print("\n🎉 All UC Python tools registered successfully!")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🔍 Step 5: Provision Databricks Vector Search Endpoint
# MAGIC Validates or provisions the Serverless Vector Search endpoint for real-time semantic search.

# COMMAND ----------
from databricks.vector_search.client import VectorSearchClient
from config import VECTOR_SEARCH_ENDPOINT
import time

vsc = VectorSearchClient()

print(f"Checking Vector Search endpoint: '{VECTOR_SEARCH_ENDPOINT}'...")
try:
    endpoint_info = vsc.get_endpoint(endpoint_name=VECTOR_SEARCH_ENDPOINT)
    print(f"✅ Vector Search endpoint '{VECTOR_SEARCH_ENDPOINT}' exists and is ready.")
except Exception as e:
    print(f"Endpoint not found. Creating standard Vector Search endpoint '{VECTOR_SEARCH_ENDPOINT}'...")
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT, endpoint_type="STANDARD")
    print(f"🚀 Provisioning triggered for endpoint '{VECTOR_SEARCH_ENDPOINT}'. Waiting for ready state...")
    time.sleep(5)
    print("✅ Provisioning initiated.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## ✅ Step 6: Final Verification
# MAGIC Query Unity Catalog to verify all created objects, tables, and functions.

# COMMAND ----------
# MAGIC %sql
# MAGIC -- Show registered functions
# MAGIC SHOW USER FUNCTIONS IN db_ai_strike_team.sec_intelligence;

# COMMAND ----------
# MAGIC %sql
# MAGIC -- Show created tables
# MAGIC SHOW TABLES IN db_ai_strike_team.sec_intelligence;

# COMMAND ----------
# MAGIC %sql
# MAGIC -- Show created volumes
# MAGIC SHOW VOLUMES IN db_ai_strike_team.sec_intelligence;

