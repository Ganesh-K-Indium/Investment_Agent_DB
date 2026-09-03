"""
Registration script for Unity Catalog governed tools.
Uses DatabricksFunctionClient from unitycatalog-ai to publish Python tools
as governed UC functions under {CATALOG}.{SCHEMA} with replace=True.
"""

import sys
import os
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DATABRICKS_CATALOG, DATABRICKS_SCHEMA
from tools.uc_tools import check_filing_status, record_feedback, get_relevant_feedback

logger = logging.getLogger("tools.register_tools")


def register_all_uc_tools(catalog: str = DATABRICKS_CATALOG, schema: str = DATABRICKS_SCHEMA):
    """
    Registers the SEC Intelligence tools into Unity Catalog via DatabricksFunctionClient.
    """
    from unitycatalog.ai.core.databricks import DatabricksFunctionClient

    logger.info("Initializing DatabricksFunctionClient for UC tool registration...")
    client = DatabricksFunctionClient()

    tools_to_register = [
        check_filing_status,
        record_feedback,
        get_relevant_feedback,
    ]

    registered_functions = []

    for tool_func in tools_to_register:
        func_name = tool_func.__name__
        logger.info("Registering function '%s' to UC %s.%s (replace=True)...", func_name, catalog, schema)
        try:
            fn_info = client.create_python_function(
                func=tool_func,
                catalog=catalog,
                schema=schema,
                replace=True,
            )
            full_name = f"{catalog}.{schema}.{func_name}"
            logger.info("Successfully registered UC function: %s", full_name)
            registered_functions.append(full_name)
        except Exception as exc:
            logger.error("Failed to register UC function '%s': %s", func_name, exc)
            raise

    return registered_functions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        registered = register_all_uc_tools()
        print(f"\n[UC TOOLS REGISTERED] Successfully registered {len(registered)} functions:")
        for r in registered:
            print(f" - {r}")
    except Exception as e:
        print(f"\n[UC TOOLS REGISTRATION FAILED] {e}", file=sys.stderr)
        sys.exit(1)

