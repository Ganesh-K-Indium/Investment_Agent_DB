"""
Streamlit Application: Databricks 2-Agent SEC Intelligence System.
Features:
1. Data Management Sidebar: Asynchronous/background SEC filing ingestion without blocking chat.
2. HITL Plan Review Toggle: Human-in-the-loop inspection and approval of query plans before execution.
3. Observability & Agent Reasoning: Real-time progress status and MLflow-tracked multi-agent execution.
4. Governed Delta Feedback Widget: Real-time user ratings and critiques persisted to Unity Catalog.
"""

import os
import sys
import json
import subprocess
import streamlit as st

# Ensure project root in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import (
    DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA,
    DATABRICKS_VOLUME,
    SERVING_ENDPOINT,
    VECTOR_SEARCH_ENDPOINT,
    VS_INDEX_NAME,
)
from tools.uc_tools import check_filing_status, record_feedback
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

@st.cache_resource
def get_supervisor_agent():
    return SECSupervisorAgent()

supervisor = get_supervisor_agent()

# ==============================================================================
# 1. Sidebar: Data Plane & Background Ingestion Controls
# ==============================================================================
with st.sidebar:
    st.title("🗄️ Ingestion Plane")
    st.caption("Decoupled Asynchronous Filing Ingestion")

    with st.expander("📥 Ingest New Filing", expanded=True):
        ingest_ticker = st.text_input("Ticker Symbol", value="NVDA", max_chars=8).upper().strip()
        ingest_form = st.selectbox("Form Type", ["10-K", "10-Q", "8-K"], index=0)
        ingest_year = st.number_input("Fiscal / Filing Year", min_value=2015, max_value=2026, value=2024, step=1)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Check Status", use_container_width=True):
                with st.spinner("Checking Delta table..."):
                    status_raw = check_filing_status(ingest_ticker, ingest_form, ingest_year)
                    status_info = json.loads(status_raw)
                    if status_info.get("status") == "INDEXED":
                        st.success(f"Indexed ({status_info.get('chunk_count')} chunks)")
                    else:
                        st.warning("Not Indexed")

        with col2:
            if st.button("Ingest & Index", type="primary", use_container_width=True):
                # Trigger jobs/ingest_sec_job.py asynchronously
                job_cmd = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "jobs", "ingest_sec_job.py"),
                    "--ticker", ingest_ticker,
                    "--form", ingest_form,
                    "--year", str(ingest_year),
                ]
                proc = subprocess.Popen(
                    job_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                job_desc = f"{ingest_ticker} {ingest_form} ({ingest_year})"
                st.session_state.background_jobs.append({
                    "target": job_desc,
                    "pid": proc.pid,
                    "process": proc,
                })
                st.info(f"Ingestion started in background for {job_desc} (PID: {proc.pid})")

    # Background Job Monitor
    if st.session_state.background_jobs:
        st.divider()
        st.subheader("⚙️ Background Tasks")
        for j in st.session_state.background_jobs:
            poll_res = j["process"].poll()
            if poll_res is None:
                st.caption(f"🔄 **{j['target']}**: Ingestion in progress (PID {j['pid']})...")
            elif poll_res == 0:
                st.caption(f"✅ **{j['target']}**: Finished successfully.")
            else:
                st.caption(f"❌ **{j['target']}**: Failed (Exit code {poll_res}).")

    st.divider()
    st.subheader("🏛️ Unity Catalog Context")
    st.text(f"Catalog: {DATABRICKS_CATALOG}")
    st.text(f"Schema:  {DATABRICKS_SCHEMA}")
    st.text(f"Volume:  {DATABRICKS_VOLUME}")
    st.text(f"Index:   {VS_INDEX_NAME.split('.')[-1]}")
    st.text(f"LLM:     {SERVING_ENDPOINT}")

# ==============================================================================
# 2. Main Interface Header & Configuration
# ==============================================================================
st.title("📈 Databricks 2-Agent SEC Intelligence")
st.markdown(
    "Production 2-Agent investment research engine with **decoupled async ingestion**, "
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
# 3. Chat History Display
# ==============================================================================
for msg_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display persistent feedback widget for assistant messages
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
                        placeholder="e.g. Always report gross margin excluding stock compensation...",
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
    # Display user query in chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if HITL plan review is enabled
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
        # Direct execution
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

