"""
Streamlit Application: Databricks 2-Agent SEC Intelligence System.
Structured into three distinct operational planes:
1. 💬 Investment Intelligence Chat: Dedicated conversation interface with Wall Street research synthesis, HITL plan review, and Delta feedback loop.
2. 📥 SEC Discovery & Ingestion Plane: Standalone data ingestion console with EDGAR discovery, accession selection, live streaming logs, and Delta table chunk verification.
3. 🩺 Diagnostics & Connection Health: Real-time validation of Databricks Token, Unity Catalog, SQL Warehouse, Vector Search Endpoint, and Model Serving.
"""

import os
import sys
import json
import time
import subprocess
from datetime import date, datetime
import streamlit as st

# Ensure project root in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import (
    DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA,
    DATABRICKS_VOLUME,
    DATABRICKS_WAREHOUSE_ID,
    CHUNKS_TABLE,
    SERVING_ENDPOINT,
    VECTOR_SEARCH_ENDPOINT,
    VS_INDEX_NAME,
    get_workspace_client,
    get_databricks_host_and_token,
)
from tools.uc_tools import check_filing_status, check_multiple_accessions_status, record_feedback
from data_pipeline.sec_loader import discover_filings_sync
from agent.supervisor import SECSupervisorAgent

st.set_page_config(
    page_title="Databricks SEC Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_plan" not in st.session_state:
    st.session_state.pending_plan = None

if "discovered_filings" not in st.session_state:
    st.session_state.discovered_filings = []

if "discovered_ticker" not in st.session_state:
    st.session_state.discovered_ticker = ""

if "trigger_ingest" not in st.session_state:
    st.session_state.trigger_ingest = None

@st.cache_resource
def get_supervisor_agent():
    return SECSupervisorAgent()

supervisor = get_supervisor_agent()

# ==============================================================================
# Sidebar: System Metadata & Quick Controls
# ==============================================================================
with st.sidebar:
    st.title("📈 SEC Intelligence")
    st.caption("Databricks Native 2-Agent System")
    st.divider()

    st.subheader("🏛️ Unity Catalog Context")
    st.markdown(f"- **Catalog**: `{DATABRICKS_CATALOG}`")
    st.markdown(f"- **Schema**: `{DATABRICKS_SCHEMA}`")
    st.markdown(f"- **Volume**: `{DATABRICKS_VOLUME}`")
    st.markdown(f"- **Warehouse**: `{DATABRICKS_WAREHOUSE_ID}`")
    st.markdown(f"- **Vector Index**: `{VS_INDEX_NAME}`")
    st.markdown(f"- **Model**: `{SERVING_ENDPOINT}`")
    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_plan = None
        st.success("Chat history cleared.")
        st.rerun()

# ==============================================================================
# Top-Level Tabs: Strict Separation Between Chat, Ingestion, and Diagnostics
# ==============================================================================
tab_chat, tab_ingest, tab_health = st.tabs([
    "💬 Investment Intelligence Chat",
    "📥 SEC Discovery & Ingestion Plane",
    "🩺 System Diagnostics & Health",
])

# ==============================================================================
# TAB 1: Investment Intelligence Chat
# ==============================================================================
with tab_chat:
    st.subheader("💬 Financial Research & Investment Briefings")
    st.caption("Ask targeted financial questions over SEC 10-K, 10-Q, and 8-K filings with verifiable chunk citations.")

    # Top Query Context Bar
    q_col1, q_col2, q_col3, q_col4 = st.columns([2, 1, 1, 2])
    with q_col1:
        target_ticker = st.text_input("Ticker Symbol", value="AAPL", key="chat_ticker").upper().strip()
    with q_col2:
        target_form = st.selectbox("Form Type", ["10-K", "10-Q", "8-K"], index=0, key="chat_form")
    with q_col3:
        target_year = st.number_input("Fiscal Year", min_value=2015, max_value=2026, value=2025, step=1, key="chat_year")
    with q_col4:
        st.write("")
        st.write("")
        hitl_review_enabled = st.checkbox(
            "Review Retrieval Plan First (HITL)",
            value=False,
            help="When checked, Supervisor Agent presents the sub-queries and analytical plan for your approval before executing vector search.",
        )

    st.divider()

    # Render Conversation Messages
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "response_id" in message:
                resp_id = message["response_id"]
                associated_ticker = message.get("ticker", target_ticker)
                associated_query = message.get("query", "")

                with st.expander("💬 Provide Feedback / Correction for this Company", expanded=False):
                    fb_col1, fb_col2 = st.columns([1, 4])
                    with fb_col1:
                        rating = st.radio(
                            "Rating",
                            ["👍 Helpful", "👎 Needs Correction"],
                            key=f"rating_{resp_id}_{msg_idx}",
                            horizontal=True,
                        )
                    with fb_col2:
                        fb_text = st.text_input(
                            "Feedback or instruction for future queries:",
                            key=f"fb_text_{resp_id}_{msg_idx}",
                            placeholder="e.g. Always highlight services gross margin and iPhone product mix...",
                        )

                    corrected_ground_truth = st.text_area(
                        "Optional corrected context / numbers:",
                        key=f"corr_text_{resp_id}_{msg_idx}",
                        placeholder="e.g. Fiscal 2025 Services revenue reached $109.2B",
                        height=68,
                    )

                    if st.button("Save Feedback to Delta Table", key=f"btn_fb_{resp_id}_{msg_idx}"):
                        if fb_text.strip() or corrected_ground_truth.strip():
                            rating_val = "POSITIVE" if "Helpful" in rating else "NEGATIVE"
                            with st.spinner("Persisting feedback to agent_feedback Delta table..."):
                                record_feedback(
                                    query=associated_query,
                                    ticker=associated_ticker,
                                    rating=rating_val,
                                    feedback_text=fb_text.strip(),
                                    corrected_context=corrected_ground_truth.strip(),
                                )
                            st.success(f"✅ Feedback recorded for {associated_ticker} in `{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.agent_feedback`!")
                        else:
                            st.warning("Please provide feedback notes or corrected context before submitting.")

    # Pending HITL Plan Approval Card
    if st.session_state.pending_plan is not None:
        pending = st.session_state.pending_plan
        st.info("### 📋 Review Retrieval Plan (Human-in-the-Loop)")
        st.markdown(f"**Target Company**: `{pending['target_ticker']}` | **Filing**: `{pending['form_type']} {pending['fiscal_year']}`")
        st.markdown(f"**Analytical Objective**: {pending.get('analytical_focus', 'N/A')}")

        st.markdown("**Planned Vector Search Sub-Queries (edit if desired):**")
        sub_queries = pending.get("planned_sub_queries", [])
        edited_queries = []
        for i, q in enumerate(sub_queries):
            eq = st.text_input(f"Sub-Query {i+1}:", value=q, key=f"hitl_query_{i}")
            edited_queries.append(eq)

        hitl_c1, hitl_c2 = st.columns([1, 4])
        with hitl_c1:
            if st.button("✅ Approve & Execute Plan", type="primary"):
                pending["planned_sub_queries"] = edited_queries
                with st.status("Executing Approved Plan with MLflow Tracing...") as status_widget:
                    status_widget.write("Dispatched to SEC Retrieval Agent...")
                    retrieval_res = supervisor.execute_retrieval(pending)
                    status_widget.write("Synthesizing financial brief...")
                    final_brief = supervisor.synthesize_report(pending["user_query"], pending, retrieval_res)
                    status_widget.update(label="Analysis Completed!", state="complete")

                resp_id = f"resp_{len(st.session_state.messages)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_brief,
                    "response_id": resp_id,
                    "ticker": pending["target_ticker"],
                    "query": pending["user_query"],
                })
                st.session_state.pending_plan = None
                st.rerun()

        with hitl_c2:
            if st.button("✕ Cancel Plan"):
                st.session_state.pending_plan = None
                st.rerun()

    # Chat Input Box
    if prompt := st.chat_input("Ask a financial question (e.g. 'What drove revenue growth and gross margin changes in fiscal 2025?'):"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if hitl_review_enabled:
            with st.status("Supervisor Designing Research Plan...") as status_box:
                status_box.write("Checking feedback memory & formulating targeted queries...")
                plan = supervisor.plan_task(
                    user_query=prompt,
                    ticker=target_ticker,
                    form_type=target_form,
                    year=target_year,
                )
                status_box.update(label="Plan Ready! Awaiting Your Approval Above.", state="complete")
            st.session_state.pending_plan = plan
            st.rerun()
        else:
            with st.chat_message("assistant"):
                with st.status("Executing 2-Agent Intelligence Workflow...") as status_box:
                    status_box.write("Step 1/3: Supervisor checking past feedback memory and designing plan...")
                    plan = supervisor.plan_task(
                        user_query=prompt,
                        ticker=target_ticker,
                        form_type=target_form,
                        year=target_year,
                    )

                    status_box.write("Step 2/3: SEC Retrieval Agent querying Vector Search index...")
                    retrieval_res = supervisor.execute_retrieval(plan)

                    status_box.write("Step 3/3: Synthesizing financial research brief with citations...")
                    final_brief = supervisor.synthesize_report(prompt, plan, retrieval_res)
                    status_box.update(label="Analysis Complete!", state="complete")

                st.markdown(final_brief)

                resp_id = f"resp_{len(st.session_state.messages)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_brief,
                    "response_id": resp_id,
                    "ticker": target_ticker,
                    "query": prompt,
                })
                st.rerun()


# ==============================================================================
# TAB 2: SEC Discovery & Ingestion Plane (Dedicated Workspace)
# ==============================================================================
with tab_ingest:
    st.subheader("📥 SEC Discovery & Ingestion Plane")
    st.caption("Discover, download from SEC EDGAR, upload to Unity Catalog Volume, and merge chunks into Delta table with Change Data Feed.")

    disc_col1, disc_col2 = st.columns([1, 2])

    with disc_col1:
        st.markdown("#### 1. Discovery Parameters")
        ingest_ticker = st.text_input("Company Ticker", value="AAPL", key="ingest_search_ticker").upper().strip()
        ingest_forms = st.multiselect("Form Types", ["10-K", "10-Q", "8-K"], default=["10-K", "10-Q"])

        filter_mode = st.radio("Discovery Filter", ["Year & Quarter", "Date Range"], horizontal=True)
        filter_year = None
        filter_quarter = None
        filter_start = None
        filter_end = None

        if filter_mode == "Year & Quarter":
            fy_col, fq_col = st.columns(2)
            with fy_col:
                use_year = st.checkbox("Filter Year", value=True)
                if use_year:
                    filter_year = st.number_input("Year", min_value=2015, max_value=2026, value=2025, step=1, key="ingest_year_val")
            with fq_col:
                q_choice = st.selectbox("Quarter", ["All Quarters", "Q1", "Q2", "Q3", "Q4"], index=0, key="ingest_q_val")
                if q_choice != "All Quarters":
                    filter_quarter = q_choice
        else:
            sd_col, ed_col = st.columns(2)
            with sd_col:
                filter_start = st.date_input("Start Date", value=date(2024, 1, 1))
            with ed_col:
                filter_end = st.date_input("End Date", value=date.today())

        if st.button("🔍 Discover SEC Filings from EDGAR", type="primary", use_container_width=True):
            with st.spinner(f"Querying SEC EDGAR for {ingest_ticker}..."):
                start_str = filter_start.strftime("%Y-%m-%d") if filter_start else None
                end_str = filter_end.strftime("%Y-%m-%d") if filter_end else None

                discovered = discover_filings_sync(
                    ticker=ingest_ticker,
                    form_types=ingest_forms,
                    year=filter_year,
                    quarter=filter_quarter,
                    start_date=start_str,
                    end_date=end_str,
                    limit=20,
                )

                if discovered:
                    accessions = [f["accession"] for f in discovered]
                    status_map = check_multiple_accessions_status(ingest_ticker, accessions)
                    for f in discovered:
                        acc = f["accession"]
                        chunks = status_map.get(acc, 0)
                        f["chunks"] = chunks
                        f["indexed"] = chunks > 0

                    st.session_state.discovered_filings = discovered
                    st.session_state.discovered_ticker = ingest_ticker
                    st.success(f"Found {len(discovered)} candidate filings for {ingest_ticker}!")
                else:
                    st.session_state.discovered_filings = []
                    st.warning(f"No filings found for {ingest_ticker} with specified filters.")

    with disc_col2:
        st.markdown(f"#### 2. Candidate Filings for **{st.session_state.discovered_ticker or ingest_ticker}**")
        if st.session_state.discovered_filings:
            filings = st.session_state.discovered_filings

            selected_accessions = []
            for idx, filing in enumerate(filings):
                f_box_col1, f_box_col2 = st.columns([1, 9])
                with f_box_col1:
                    is_sel = st.checkbox("", value=(not filing["indexed"]), key=f"sel_{filing['accession']}_{idx}")
                    if is_sel:
                        selected_accessions.append(filing["accession"])
                with f_box_col2:
                    status_badge = f"🟢 Indexed ({filing['chunks']} chunks)" if filing["indexed"] else "⚪ Not Ingested"
                    st.markdown(f"**{filing['form']}** ({filing['filing_date']}) — Quarter: `{filing.get('quarter', 'N/A')}` | `{filing['accession']}` | {status_badge}")

            st.divider()
            b_c1, b_c2 = st.columns(2)
            with b_c1:
                if st.button(f"⚡ Ingest Selected ({len(selected_accessions)})", type="primary", use_container_width=True, disabled=(len(selected_accessions) == 0)):
                    st.session_state.trigger_ingest = {
                        "ticker": st.session_state.discovered_ticker,
                        "accessions": selected_accessions,
                        "desc": f"{len(selected_accessions)} selected filings for {st.session_state.discovered_ticker}",
                    }
                    st.rerun()

            with b_c2:
                all_accs = [f["accession"] for f in filings]
                if st.button(f"📥 Ingest All ({len(all_accs)})", use_container_width=True):
                    st.session_state.trigger_ingest = {
                        "ticker": st.session_state.discovered_ticker,
                        "accessions": all_accs,
                        "desc": f"all {len(all_accs)} candidate filings for {st.session_state.discovered_ticker}",
                    }
                    st.rerun()
        else:
            st.info("Click **'🔍 Discover SEC Filings from EDGAR'** to search and view available filings.")

    # Dedicated Live Ingestion Terminal Console (In Tab 2 only!)
    if st.session_state.trigger_ingest:
        ingest_job = st.session_state.trigger_ingest
        st.divider()
        st.markdown(f"### ⚡ Live Ingestion Console: **{ingest_job['desc']}**")

        with st.status(f"Running Ingestion Pipeline for {ingest_job['desc']}...", expanded=True) as status_box:
            st.write("🚀 Initializing EDGAR download, Unity Catalog Volume upload, and Delta MERGE...")
            log_box = st.empty()
            captured_lines = []

            job_cmd = [
                sys.executable,
                "-u",
                os.path.join(os.path.dirname(__file__), "jobs", "ingest_sec_job.py"),
                "--ticker", ingest_job["ticker"],
                "--accessions", *ingest_job["accessions"],
            ]

            env_copy = os.environ.copy()
            proc = subprocess.Popen(
                job_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env_copy,
            )

            for line in proc.stdout:
                captured_lines.append(line)
                tail = "".join(captured_lines[-15:])
                log_box.code(tail, language="text")

            proc.wait()

            if proc.returncode == 0:
                status_box.update(label=f"✅ Ingestion Completed Successfully for {ingest_job['desc']}!", state="complete", expanded=False)
                st.success("🎉 **Success!** Filing data stored in Unity Catalog Volume and merged into Delta table.")

                # Real-time post-ingestion Delta verification check
                try:
                    verified_map = check_multiple_accessions_status(ingest_job["ticker"], ingest_job["accessions"])
                    total_verified = sum(verified_map.values())
                    st.info(f"📊 **Delta Verification Confirmed**: `{total_verified}` total chunks confirmed in `{CHUNKS_TABLE}`.")

                    for f in st.session_state.discovered_filings:
                        if f["accession"] in verified_map:
                            f["chunks"] = verified_map[f["accession"]]
                            f["indexed"] = f["chunks"] > 0
                except Exception as v_err:
                    st.warning(f"Delta verification notice: {v_err}")
            else:
                status_box.update(label=f"❌ Ingestion Failed (Exit code {proc.returncode})", state="error", expanded=True)
                st.error("Ingestion encountered an error. Full logs below:")
                log_box.code("".join(captured_lines), language="text")

        with st.expander("📜 Full Ingestion Terminal Log", expanded=(proc.returncode != 0)):
            st.code("".join(captured_lines), language="text")

        if st.button("✕ Close Ingestion Console", use_container_width=True):
            st.session_state.trigger_ingest = None
            st.rerun()


# ==============================================================================
# TAB 3: Diagnostics & Connection Health
# ==============================================================================
with tab_health:
    st.subheader("🩺 System Diagnostics & Connection Health")
    st.caption("Verify live Databricks authentication, Unity Catalog access, SQL Warehouse latency, and Vector Search status.")

    if st.button("🔄 Run Live System Diagnostics", type="primary"):
        with st.spinner("Testing all Databricks connections and permissions..."):
            diag_results = {}

            # 1. Host & Token Check
            host, token = get_databricks_host_and_token()
            token_masked = f"{token[:8]}...{token[-4:]}" if token and len(token) > 12 else "NOT FOUND / NO-TOKEN"
            diag_results["auth"] = {
                "host": host,
                "token_detected": bool(token and token != "no-token"),
                "token_preview": token_masked,
            }

            # 2. Workspace Client
            w = None
            try:
                w = get_workspace_client()
                user_info = w.current_user.me()
                diag_results["user"] = getattr(user_info, "user_name", "Unknown")
                diag_results["client_ok"] = True
            except Exception as e:
                diag_results["user"] = f"Error: {e}"
                diag_results["client_ok"] = False

            # 3. SQL Warehouse Delta Table Check
            try:
                t0 = time.time()
                stmt = f"SELECT COUNT(*) FROM {CHUNKS_TABLE};"
                resp = w.statement_execution.execute_statement(
                    warehouse_id=DATABRICKS_WAREHOUSE_ID,
                    statement=stmt,
                    wait_timeout="15s",
                )
                latency = round(time.time() - t0, 2)
                row_count = resp.result.data_array[0][0] if resp.result and resp.result.data_array else "0"
                diag_results["warehouse"] = {"status": "ONLINE", "latency": f"{latency}s", "rows": row_count}
            except Exception as e:
                diag_results["warehouse"] = {"status": "ERROR", "detail": str(e)}

            # 4. Vector Search Index Check
            try:
                from data_pipeline.vector_indexer import get_vector_search_client
                vsc = get_vector_search_client()
                idx = vsc.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT, index_name=VS_INDEX_NAME)
                idx_desc = idx.describe()
                status_dict = idx_desc.get("status", {})
                state = status_dict.get("detailed_state") or status_dict.get("state") or "UNKNOWN"
                diag_results["vector_search"] = {
                    "status": state,
                    "endpoint": VECTOR_SEARCH_ENDPOINT,
                    "index": VS_INDEX_NAME,
                    "raw": status_dict,
                }
            except Exception as e:
                diag_results["vector_search"] = {"status": "ERROR", "detail": str(e)}

            # Display Results Cards
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🔑 Identity & Authentication")
                st.markdown(f"- **Workspace Host**: `{diag_results['auth']['host']}`")
                st.markdown(f"- **Active User Identity**: `{diag_results['user']}`")
                st.markdown(f"- **Token Active**: `{'✅ Yes' if diag_results['auth']['token_detected'] else '❌ No'}` (`{diag_results['auth']['token_preview']}`)")

                st.markdown("#### 🏛️ SQL Warehouse & Delta Table")
                wh = diag_results["warehouse"]
                if wh["status"] == "ONLINE":
                    st.success(f"✅ Warehouse `{DATABRICKS_WAREHOUSE_ID}` online ({wh['latency']}) — `{wh['rows']}` total chunks in `{CHUNKS_TABLE}`.")
                else:
                    st.error(f"❌ SQL Warehouse Error: `{wh.get('detail')}`")

            with c2:
                st.markdown("#### 🔍 Vector Search Index")
                vs = diag_results["vector_search"]
                if vs["status"] == "ONLINE":
                    st.success(f"✅ Vector Index `{VS_INDEX_NAME}` is **ONLINE** and ready for similarity search!")
                elif "PROVISIONING" in vs["status"]:
                    st.warning(f"⏳ Vector Index is currently **{vs['status']}**. Please wait 2-3 minutes for initial embedding sync.")
                else:
                    st.error(f"❌ Vector Search Issue: `{vs.get('detail', vs['status'])}`")
                if "raw" in vs:
                    st.caption(f"Raw State: `{vs['raw']}`")
    else:
        st.info("Click **'🔄 Run Live System Diagnostics'** above to test your connections in real time.")
