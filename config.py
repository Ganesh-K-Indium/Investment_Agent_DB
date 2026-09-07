"""
Configuration module for the Databricks SEC Intelligence Agent System.
Sources settings from environment variables with production defaults.
Configures MLflow tracing and silences noisy library loggers.
"""

import os
import logging

# ==============================================================================
# 1. Environment Configurations & Production Defaults
# ==============================================================================
DATABRICKS_CATALOG = os.getenv("DATABRICKS_CATALOG", "db_ai_strike_team")
DATABRICKS_SCHEMA = os.getenv("DATABRICKS_SCHEMA", "sec_intelligence")
DATABRICKS_VOLUME = os.getenv("DATABRICKS_VOLUME", "raw_filings")
DATABRICKS_WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "58fc0be4c55c7e10")
VECTOR_SEARCH_ENDPOINT = os.getenv("VECTOR_SEARCH_ENDPOINT", "sec_vs_endpoint")
SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct")
EMBEDDING_MODEL_ENDPOINT = os.getenv("EMBEDDING_MODEL_ENDPOINT", "databricks-bge-large-en")

# Derived Unity Catalog Paths
FULL_SCHEMA = f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}"
VOLUME_PATH = f"/Volumes/{DATABRICKS_CATALOG}/{DATABRICKS_SCHEMA}/{DATABRICKS_VOLUME}"
LOCAL_FALLBACK_VOLUME = os.getenv("LOCAL_FALLBACK_VOLUME", os.path.join(os.getcwd(), "raw_filings"))

CHUNKS_TABLE = f"{FULL_SCHEMA}.sec_filing_chunks"
FEEDBACK_TABLE = f"{FULL_SCHEMA}.agent_feedback"
VS_INDEX_NAME = f"{FULL_SCHEMA}.sec_filing_index"

# SEC User Agent
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "Indium Capital contact@indium.com")
SEC_REQUEST_RATE_LIMIT = int(os.getenv("SEC_REQUEST_RATE_LIMIT", "10"))

# ==============================================================================
# 2. Logging and Observability Configuration
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Silence noisy HTTP and SDK loggers in favor of native MLflow tracing
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Initialize MLflow OpenAI autologging safely
try:
    import mlflow
    if os.getenv("DATABRICKS_HOST"):
        mlflow.set_tracking_uri("databricks")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    if hasattr(mlflow, "openai") and hasattr(mlflow.openai, "autolog"):
        mlflow.openai.autolog()
except Exception as e:
    logging.getLogger(__name__).warning("MLflow autolog could not be initialized at config load: %s", e)


def get_workspace_client():
    """
    Returns a WorkspaceClient instance.
    If DATABRICKS_TOKEN is provided, prioritizes PAT authentication (acting as the user).
    Otherwise falls back to ambient Databricks authentication (OAuth M2M service principal).
    """
    from databricks.sdk import WorkspaceClient

    token = os.getenv("DATABRICKS_TOKEN")
    host = os.getenv("DATABRICKS_HOST")
    if token and host:
        return WorkspaceClient(host=host, token=token)
    elif token:
        return WorkspaceClient(token=token)
    return WorkspaceClient()


def get_databricks_host_and_token():
    """
    Resolves the active Databricks host and valid Bearer token across:
    1. Environment variables (DATABRICKS_HOST, DATABRICKS_TOKEN)
    2. WorkspaceClient OAuth / M2M authentication (w.config.authenticate())
    3. Databricks Notebook DBUtils context
    """
    token = os.getenv("DATABRICKS_TOKEN")
    host = os.getenv("DATABRICKS_HOST")

    w = None
    try:
        w = get_workspace_client()
        if not host:
            host = getattr(w.config, "host", None)
        if not token:
            auth_headers = w.config.authenticate()
            if auth_headers and "Authorization" in auth_headers:
                token = auth_headers["Authorization"].replace("Bearer ", "").strip()
    except Exception:
        pass

    # Notebook context fallback if token is still empty
    if not token:
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark:
                from pyspark.dbutils import DBUtils
                dbutils = DBUtils(spark)
                token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
                if not host:
                    host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
        except Exception:
            pass

    host = (host or "https://databricks.local").rstrip("/")
    return host, (token or "no-token")

