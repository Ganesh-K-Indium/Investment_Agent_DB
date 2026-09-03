"""
Streamlit Application: Databricks 2-Agent SEC Intelligence System.
Features:
1. Granular Ingestion Plane:
   - Filter by Form Type (10-K, 10-Q, 8-K), Fiscal Year, Quarter (Q1-Q4), or Date Range.
   - Live SEC EDGAR Discovery & Index Status Check.
   - Interactive Multi-Select / Ingest Specific Filings or Ingest All.
   - Deduplication Prevention (Delta MERGE) & Error Isolation.
2. Online 2-Agent Intelligence:
   - HITL Plan Review Toggle: Review and modify retrieval plan before execution.
   - Multi-Perspective Vector Retrieval with Deduplication.
   - Governed Unity Catalog Feedback Widget persisting to agent_feedback Delta table.
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
    CHUNKS_TABLE,
    SERVING_ENDPOINT,
    VS_INDEX_NAME,
)
from tools.uc_tools import check_filing_status, check_multiple_accessions_status, record_feedback
from data_pipeline.sec_loader import discover_filings_sync
from agent.supervisor import SECSupervisorAgent

st.set_page_config(
    page_title="Databricks SEC Intelligence",
    page_icon="📈",
    layout="wide",
)

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_plan" not in st.session_state:
    st.session_state.pending_plan = None

if "background_jobs" not in st.session_state:
    st.session_state.background_jobs = []

if "discovered_filings" not in st.session_state:
    st.session_state.discovered_filings = []

if "discovered_ticker" not in st.session_state:
    st.session_state.discovered_ticker = ""

@st.cache_resource
def get_supervisor_agent():
    return SECSupervisorAgent()

supervisor = get_supervisor_agent()

# ==============================================================================
# 1. Sidebar: Granular Ingestion Plane & Discovery Controls
# ==============================================================================
with st.sidebar:
    st.title("🗄️ Ingestion Plane")
    st.caption("Granular SEC Discovery & Asynchronous Ingestion")

    with st.expander("🔍 Step 1: Discover Filings", expanded=True):
        search_ticker = st.text_input("Ticker Symbol", value="NVDA", max_chars=8).upper().strip()
        
        selected_forms = st.multiselect(
            "Form Types",
            options=["10-K", "10-Q", "8-K"],
            default=["10-K", "10-Q"],
        )

        filter_mode = st.radio("Filter Mode", ["Year & Quarter", "Date Range"], horizontal=True)

        filter_year = None
        filter_quarter = None
        filter_start_date = None
        filter_end_date = None

        if filter_mode == "Year & Quarter":
            c_y, c_q = st.columns(2)
            with c_y:
                use_year = st.checkbox("Filter Year", value=True)
                if use_year:
                    filter_year = st.number_input("Year", min_value=2015, max_value=2026, value=2024, step=1)
            with c_q:
                q_choice = st.selectbox("Quarter", ["All Quarters", "Q1", "Q2", "Q3", "Q4"], index=0)
                if q_choice != "All Quarters":
                    filter_quarter = q_choice
        else:
            c_s, c_e = st.columns(2)
            with c_s:
                filter_start_date = st.date_input("Start Date", value=date(2024, 1, 1))
            with c_e:
                filter_end_date = st.date_input("End Date", value=date.today())

        if st.button("🔎 Discover Filings on SEC EDGAR", type="primary", use_container_width=True):
            with st.spinner(f"Querying SEC EDGAR for {search_ticker}..."):
                try:
                    filings = discover_filings_sync(
                        ticker=search_ticker,
                        form_types=selected_forms if selected_forms else None,
                        year=filter_year,
                        quarter=filter_quarter,
                        start_date=str(filter_start_date) if filter_start_date else None,
                        end_date=str(filter_end_date) if filter_end_date else None,
                    )
                    # Check existing Delta index status for all discovered accessions
                    acc_list = [f["accession"] for f in filings]
                    status_map = check_multiple_accessions_status(search_ticker, acc_list)
                    for f in filings:
                        f["chunks"] = status_map.get(f["accession"], 0)
                        f["indexed"] = f["chunks"] > 0

                    st.session_state.discovered_filings = filings
                    st.session_state.discovered_ticker = search_ticker
                    st.success(f"Found {len(filings)} filing(s) for {search_ticker}!")
                except Exception as e:
                    st.error(f"Discovery error: {e}")

    # Step 2: Selective / Batch Ingestion
    if st.session_state.discovered_filings:
        with st.expander(f"📥 Step 2: Select & Ingest ({st.session_state.discovered_ticker})", expanded=True):
            discovered = st.session_state.discovered_filings
            st.caption(f"Select filings to ingest into `{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}`:")

            selected_accessions = []
            select_all = st.checkbox("Select All Discovered", value=False)

            for idx, item in enumerate(discovered):
                status_icon = "✅ Indexed" if item.get("indexed") else "⚪ Ready"
                label = f"{item['form']} ({item.get('quarter', 'N/A')} {item['year']}) | {item['filing_date']} | {status_icon}"
                is_checked = select_all
                c_box = st.checkbox(label, value=is_checked, key=f"f_chk_{item['accession']}_{idx}")
                if c_box:
                    selected_accessions.append(item["accession"])

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📥 Ingest Selected", use_container_width=True, disabled=(len(selected_accessions) == 0)):
                    desc = f"{st.session_state.discovered_ticker} ({len(selected_accessions)} selected filings)"
                    st.session_state.trigger_ingest = {
                        "ticker": st.session_state.discovered_ticker,
                        "accessions": selected_accessions,
                        "desc": desc,
                    }
                    st.rerun()

            with col_btn2:
                if st.button("⚡ Ingest All", type="secondary", use_container_width=True):
                    all_accs = [f["accession"] for f in discovered]
                    desc = f"{st.session_state.discovered_ticker} (All {len(all_accs)} filings)"
                    st.session_state.trigger_ingest = {
                        "ticker": st.session_state.discovered_ticker,
                        "accessions": all_accs,
                        "desc": desc,
                    }
                    st.rerun()

    # Background Job Monitor & Error Recovery
    if st.session_state.background_jobs:
        st.divider()
        st.subheader("⚙️ Background Ingestion Monitor")
        for j in st.session_state.background_jobs:
            poll_res = j["process"].poll()
            if poll_res is None:
                st.caption(f"🔄 **{j['target']}**: In progress (PID {j['pid']})...")
            elif poll_res == 0:
                st.caption(f"✅ **{j['target']}**: Finished successfully.")
            else:
                st.caption(f"❌ **{j['target']}**: Exited with code {poll_res}.")

    st.divider()
    st.subheader("🏛️ Unity Catalog Info")
    st.caption(f"**Catalog:** `{DATABRICKS_CATALOG}`")
    st.caption(f"**Schema:** `{DATABRICKS_SCHEMA}`")
    st.caption(f"**Volume:** `{DATABRICKS_VOLUME}`")
    st.caption(f"**Deduplication:** Delta MERGE on `chunk_id`")

# ==============================================================================
# 2. Main Interface Header & Configuration
# ==============================================================================
st.title("📈 Databricks 2-Agent SEC Intelligence")
st.markdown(
    "Databricks-native investment intelligence with **decoupled granular ingestion**, "
    "**multi-perspective retrieval**, and **Delta-backed HITL memory**."
)

col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
with col_ctrl1:
    target_ticker = st.text_input("Active Query Ticker", value="NVDA", key="active_ticker").upper().strip()
with col_ctrl2:
    target_form = st.selectbox("Form", ["10-K", "10-Q", "8-K"], index=0, key="active_form")
with col_ctrl3:
    target_year = st.number_input("Year", min_value=2015, max_value=2026, value=2024, step=1, key="active_year")

hitl_review_enabled = st.checkbox(
    "Review Agent Retrieval Plan Before Execution (HITL Plan Review)",
    value=False,
    help="When checked, Supervisor Agent will generate and present the research plan for your approval before executing vector search.",
)

# ==============================================================================
# Live Ingestion & Verification Center (Full-Width Main Panel)
# ==============================================================================
if "trigger_ingest" in st.session_state and st.session_state.trigger_ingest:
    ingest_info = st.session_state.trigger_ingest
    st.divider()
    st.markdown(f"### 📥 Ingestion Console: **{ingest_info['desc']}**")

    with st.status(f"⚡ Processing ingestion for {ingest_info['desc']}...", expanded=True) as status_box:
        st.write("🚀 Starting Databricks ingestion pipeline...")
        log_box = st.empty()
        captured_lines = []

        job_cmd = [
            sys.executable,
            "-u",
            os.path.join(os.path.dirname(__file__), "jobs", "ingest_sec_job.py"),
            "--ticker", ingest_info["ticker"],
            "--accessions", *ingest_info["accessions"],
        ]

        proc = subprocess.Popen(
            job_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            captured_lines.append(line)
            tail = "".join(captured_lines[-15:])
            log_box.code(tail, language="text")

        proc.wait()

        if proc.returncode == 0:
            status_box.update(label=f"✅ Ingestion Succeeded for {ingest_info['desc']}!", state="complete", expanded=False)
            st.success(f"🎉 **Ingestion Succeeded!** Filings saved to Unity Catalog Volume `{DATABRICKS_VOLUME}` and indexed into Delta table `{CHUNKS_TABLE}`.")

            # Post-Ingestion Live Verification Check
            try:
                verified_map = check_multiple_accessions_status(ingest_info["ticker"], ingest_info["accessions"])
                total_verified = sum(verified_map.values())
                st.info(f"📊 **Delta Verification Confirmed**: `{total_verified}` total chunks confirmed in `{CHUNKS_TABLE}`.")

                # Update discovered filings cache with new indexed status
                for f in st.session_state.discovered_filings:
                    if f["accession"] in verified_map:
                        f["chunks"] = verified_map[f["accession"]]
                        f["indexed"] = f["chunks"] > 0
            except Exception as v_err:
                st.warning(f"Verification notice: {v_err}")
        else:
            status_box.update(label=f"❌ Ingestion Failed for {ingest_info['desc']} (Exit code {proc.returncode})", state="error", expanded=True)
            st.error("Ingestion job encountered an error. Full output below:")
            log_box.code("".join(captured_lines), language="text")

    with st.expander("📜 Full Ingestion Console Log", expanded=(proc.returncode != 0)):
        st.code("".join(captured_lines), language="text")

    if st.button("✕ Close Ingestion Console", use_container_width=True):
        del st.session_state["trigger_ingest"]
        st.rerun()

    st.divider()
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
                        "Tell the agent what to correct or remember for this company:",
                        key=f"fb_text_{resp_id}_{msg_idx}",
                        placeholder="e.g. Always report gross margin excluding stock-based compensation...",
                    )
                
                corrected_ground_truth = st.text_area(
                    "Optional corrected ground truth / figure:",
                    key=f"corr_text_{resp_id}_{msg_idx}",
                    placeholder="e.g. Data Center revenue was $18.4B in Q4 2024",
                    height=68,
                )

                if st.button("Save Feedback to Unity Catalog Delta Table", key=f"btn_fb_{resp_id}_{msg_idx}"):
                    if fb_text.strip() or corrected_ground_truth.strip():
                        rating_val = "POSITIVE" if "Helpful" in rating else "NEGATIVE"
                        with st.spinner("Persisting feedback to Delta memory..."):
                            record_feedback(
                                query=associated_query,
                                ticker=associated_ticker,
                                rating=rating_val,
                                feedback_text=fb_text.strip(),
                                corrected_context=corrected_ground_truth.strip(),
                            )
                        st.success(f"Feedback saved for {associated_ticker} in `{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.agent_feedback`!")
                    else:
                        st.warning("Please provide feedback text or corrected figures before submitting.")

# ==============================================================================
# 4. Interactive HITL Plan Approval Modal/Card (if pending)
# ==============================================================================
if st.session_state.pending_plan is not None:
    pending = st.session_state.pending_plan
    st.info("### 📋 Human-in-the-Loop: Review Retrieval Plan")
    st.markdown(f"**Target Company**: `{pending['target_ticker']}` | **Filing**: `{pending['form_type']} {pending['fiscal_year']}`")
    st.markdown(f"**Analytical Focus**: {pending.get('analytical_focus', 'N/A')}")
    
    st.markdown("**Planned Vector Search Sub-Queries:**")
    sub_queries = pending.get("planned_sub_queries", [])
    edited_queries = []
    for i, q in enumerate(sub_queries):
        eq = st.text_input(f"Sub-Query {i+1}:", value=q, key=f"hitl_query_{i}")
        edited_queries.append(eq)

    hitl_c1, hitl_c2 = st.columns([1, 4])
    with hitl_c1:
        if st.button("✅ Approve & Run", type="primary"):
            pending["planned_sub_queries"] = edited_queries
            with st.status("Executing Approved Plan with MLflow Tracing...") as status_widget:
                status_widget.write("Dispatched to Intelligent SEC Retrieval Agent...")
                retrieval_res = supervisor.execute_retrieval(pending)
                status_widget.write("Synthesizing Wall Street research brief...")
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
        if st.button("Cancel Plan"):
            st.session_state.pending_plan = None
            st.rerun()

# ==============================================================================
# 5. User Chat Input Handling
# ==============================================================================
if prompt := st.chat_input("Ask an investment question (e.g. 'What drove gross margin changes in Q4?'):"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if hitl_review_enabled:
        with st.status("Supervisor Planning & Memory Retrieval...") as status_box:
            status_box.write("Checking Delta memory for past company feedback...")
            plan = supervisor.plan_task(
                user_query=prompt,
                ticker=target_ticker,
                form_type=target_form,
                year=target_year,
            )
            status_box.update(label="Research Plan Generated! Awaiting User Approval.", state="complete")
        st.session_state.pending_plan = plan
        st.rerun()
    else:
        with st.chat_message("assistant"):
            with st.status("Executing 2-Agent Intelligence Workflow...") as status_box:
                status_box.write("Step 1/3: Supervisor retrieving past feedback memory and designing plan...")
                plan = supervisor.plan_task(
                    user_query=prompt,
                    ticker=target_ticker,
                    form_type=target_form,
                    year=target_year,
                )

                status_box.write("Step 2/3: SEC Retrieval Agent checking filing status and executing multi-query vector search...")
                retrieval_res = supervisor.execute_retrieval(plan)

                status_box.write("Step 3/3: Synthesizing financial research brief with citations...")
                final_brief = supervisor.synthesize_report(prompt, plan, retrieval_res)
                status_box.update(label="Intelligence Workflow Complete!", state="complete")

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
