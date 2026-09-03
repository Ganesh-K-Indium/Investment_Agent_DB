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
DATABRICKS_CATALOG = os.getenv("DATABRICKS_CATALOG", "investment_prod")
DATABRICKS_SCHEMA = os.getenv("DATABRICKS_SCHEMA", "sec_intelligence")
DATABRICKS_VOLUME = os.getenv("DATABRICKS_VOLUME", "raw_filings")
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
