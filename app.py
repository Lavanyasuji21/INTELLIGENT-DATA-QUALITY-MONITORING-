import os
import sys
import uuid
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Add base directory to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.core_engine import DataQualityEngine
from backend.sample_data import (
    generate_sample_quality_dataset,
    generate_sample_drift_datasets
)

# ================================================================
# PAGE CONFIGURATION & CUSTOM SAAS STYLING
# ================================================================

st.set_page_config(
    page_title="Intelligent Data Quality Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS-style Light Theme CSS
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Top header */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .main-header h1 {
        margin: 0;
        font-size: 26px;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #94a3b8;
    }
    
    /* Metric Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.06);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
    }
    .kpi-subtext {
        font-size: 12px;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-teal { background-color: #ccfbf1; color: #0f766e; }
    .badge-blue { background-color: #dbeafe; color: #1d4ed8; }
    .badge-green { background-color: #dcfce7; color: #15803d; }
    .badge-amber { background-color: #fef3c7; color: #b45309; }
    .badge-red { background-color: #fee2e2; color: #b91c1c; }
    .badge-gray { background-color: #f1f5f9; color: #475569; }

    /* Info Callout */
    .callout-box {
        background-color: #ffffff;
        border-left: 4px solid #0d9488;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    
    /* Empty State Card */
    .empty-state {
        background: #ffffff;
        border: 2px dashed #cbd5e1;
        border-radius: 14px;
        padding: 45px 30px;
        text-align: center;
        margin: 20px 0;
    }
    .empty-state h3 {
        color: #1e293b;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .empty-state p {
        color: #64748b;
        font-size: 14px;
        max-width: 580px;
        margin: 0 auto 20px auto;
        line-height: 1.5;
    }
    
    /* Buttons and tabs polish */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.15s ease;
    }
    
    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Dataframe tables */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        background: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================
# SESSION STATE INITIALIZATION (MULTI-USER ISOLATION)
# ================================================================

if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8]

if "engine" not in st.session_state:
    # Use temporary directory or project reports directory safely per session
    session_report_dir = os.path.join(tempfile.gettempdir(), f"dqm_reports_{st.session_state['session_id']}")
    os.makedirs(session_report_dir, exist_ok=True)
    st.session_state["engine"] = DataQualityEngine(
        session_id=st.session_state["session_id"],
        base_report_dir=session_report_dir
    )

engine: DataQualityEngine = st.session_state["engine"]

if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = None

if "approval_message" not in st.session_state:
    st.session_state["approval_message"] = None

if "duplicate_message" not in st.session_state:
    st.session_state["duplicate_message"] = None

if "override_message" not in st.session_state:
    st.session_state["override_message"] = None


# Helper to safely read file bytes for download buttons
def get_file_bytes(file_path):
    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return f.read()
    return None


# ================================================================
# SIDEBAR NAVIGATION & SYSTEM STATUS
# ================================================================

with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 16px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 16px;">
        <div style="font-size: 20px; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 24px;">🛡️</span> DataQuality AI
        </div>
        <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
            Intelligent Data Quality & Drift Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊 Overview Dashboard",
            "🔍 Data Quality Analysis",
            "✍️ Human Approval & Override",
            "🧠 Anomaly Detection",
            "📈 Data Drift Analysis",
            "⚖️ Quality Comparison",
            "📑 Combined Reports"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Session Status Widget
    st.markdown("<div style='font-size: 12px; font-weight: 700; color: #64748b; margin-bottom: 10px; text-transform: uppercase;'>Session Status</div>", unsafe_allow_html=True)

    # Status indicators
    has_dataset = engine.APP["original"] is not None
    is_approved = engine.APP["approved"] is not None
    has_anomaly = engine.APP["anomaly_report"] is not None
    has_drift = engine.APP["drift_report"] is not None
    has_comparison = engine.APP["quality_comparison"] is not None

    status_html = f"""
    <div style="font-size: 13px; color: #334155; line-height: 1.9;">
        <div><b>Session ID:</b> <span class="badge badge-gray">{st.session_state['session_id']}</span></div>
        <div><b>Active Dataset:</b> {f'<span class="badge badge-teal">{st.session_state["dataset_name"]}</span>' if has_dataset else '<span class="badge badge-gray">None</span>'}</div>
        <div><b>Imputation:</b> {'<span class="badge badge-green">Ready</span>' if has_dataset else '<span class="badge badge-gray">Pending</span>'}</div>
        <div><b>Anomaly ML:</b> {'<span class="badge badge-green">Complete</span>' if has_anomaly else '<span class="badge badge-gray">Not run</span>'}</div>
        <div><b>Data Drift:</b> {'<span class="badge badge-green">Complete</span>' if has_drift else '<span class="badge badge-gray">Not run</span>'}</div>
        <div><b>Comparison:</b> {'<span class="badge badge-green">Complete</span>' if has_comparison else '<span class="badge badge-gray">Not run</span>'}</div>
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Reset Active Session", use_container_width=True):
        st.session_state["session_id"] = str(uuid.uuid4())[:8]
        session_report_dir = os.path.join(tempfile.gettempdir(), f"dqm_reports_{st.session_state['session_id']}")
        os.makedirs(session_report_dir, exist_ok=True)
        st.session_state["engine"] = DataQualityEngine(
            session_id=st.session_state["session_id"],
            base_report_dir=session_report_dir
        )
        st.session_state["dataset_name"] = None
        st.session_state["approval_message"] = None
        st.session_state["duplicate_message"] = None
        st.session_state["override_message"] = None
        st.rerun()

    st.markdown("""
    <div style="font-size: 11px; color: #94a3b8; text-align: center; margin-top: 30px;">
        Enterprise Data Quality Monitor v2.0<br/>
        Deployable on Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)


# ================================================================
# PAGE 1: OVERVIEW DASHBOARD
# ================================================================

if page == "📊 Overview Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Quality Monitoring Overview</h1>
        <p>Real-time analytics, automated data profiling, machine learning anomaly detection, and drift governance.</p>
    </div>
    """, unsafe_allow_html=True)

    if engine.APP["original"] is not None:
        metrics = engine.calculate_quality_metrics(engine.APP["original"])
        appr_metrics = engine.calculate_quality_metrics(engine.APP["approved"]) if engine.APP["approved"] is not None else metrics

        # Top 6 KPI Metric Cards
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Total Rows</div>
                <div class="kpi-value">{metrics['Rows']:,}</div>
                <div class="kpi-subtext">Source dataset</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Columns</div>
                <div class="kpi-value">{metrics['Columns']:,}</div>
                <div class="kpi-subtext">Features analyzed</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Missing Cells</div>
                <div class="kpi-value" style="color: {'#ef4444' if metrics['Missing Cells'] > 0 else '#10b981'};">{metrics['Missing Cells']:,}</div>
                <div class="kpi-subtext">{'Needs imputation' if metrics['Missing Cells'] > 0 else 'Clean'}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Duplicates</div>
                <div class="kpi-value" style="color: {'#f59e0b' if metrics['Duplicate Rows'] > 0 else '#10b981'};">{metrics['Duplicate Rows']:,}</div>
                <div class="kpi-subtext">Repeated records</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            q_score = metrics['Quality Score (%)']
            score_color = '#10b981' if q_score >= 85 else ('#f59e0b' if q_score >= 65 else '#ef4444')
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Quality Score</div>
                <div class="kpi-value" style="color: {score_color};">{q_score}%</div>
                <div class="kpi-subtext">Initial rating</div>
            </div>
            """, unsafe_allow_html=True)

        with col6:
            if engine.APP["anomaly_report"] is not None:
                anom_count = int(engine.APP["anomaly_report"]["Either_Model_Flagged"].sum())
                anom_text = f"{anom_count:,}"
                anom_color = "#ef4444" if anom_count > 0 else "#10b981"
            else:
                anom_text = "Pending"
                anom_color = "#64748b"

            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Anomalies</div>
                <div class="kpi-value" style="color: {anom_color};">{anom_text}</div>
                <div class="kpi-subtext">{'Detected by ML' if engine.APP['anomaly_report'] is not None else 'Not analyzed'}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

        # Overview Grid
        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.markdown("### 📋 Detected Quality Issues Summary")
            if engine.APP["issues"] is not None and not engine.APP["issues"].empty:
                st.dataframe(
                    engine.APP["issues"],
                    use_container_width=True,
                    height=280
                )
            else:
                st.success("🎉 No significant quality issues detected in the uploaded dataset!")

            if engine.APP["imputation_report"] is not None and not engine.APP["imputation_report"].empty:
                st.markdown("### 🌲 Random Forest Imputation Pipeline")
                st.dataframe(
                    engine.APP["imputation_report"],
                    use_container_width=True,
                    height=200
                )

        with right_col:
            st.markdown("### 🛡️ Analysis & Governance Pipeline")

            # Governance Cards
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; color: #1e293b;">1. Data Quality & Profiling</div>
                    <span class="badge badge-green">Completed</span>
                </div>
                <div style="font-size: 13px; color: #64748b; margin-top: 4px;">
                    {len(engine.APP['issues'])} issues flagged. Initial score: {metrics['Quality Score (%)']}%.
                </div>
            </div>

            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; color: #1e293b;">2. Human Approval & Override</div>
                    <span class="badge {'badge-teal' if engine.APP['approved'] is not None else 'badge-amber'}">
                        {'Ready / Modified' if engine.APP['approved'] is not None else 'Awaiting Decision'}
                    </span>
                </div>
                <div style="font-size: 13px; color: #64748b; margin-top: 4px;">
                    Approved rows: {appr_metrics['Rows']:,}. Quality score: {appr_metrics['Quality Score (%)']}%.
                </div>
            </div>

            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; color: #1e293b;">3. Deep Learning Anomaly Detection</div>
                    <span class="badge {'badge-green' if engine.APP['anomaly_report'] is not None else 'badge-gray'}">
                        {'Completed' if engine.APP['anomaly_report'] is not None else 'Not Run Yet'}
                    </span>
                </div>
                <div style="font-size: 13px; color: #64748b; margin-top: 4px;">
                    Autoencoder + Isolation Forest dual consensus model.
                </div>
            </div>

            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 600; color: #1e293b;">4. Population Drift Detection</div>
                    <span class="badge {'badge-green' if engine.APP['drift_report'] is not None else 'badge-gray'}">
                        {'Completed' if engine.APP['drift_report'] is not None else 'Not Run Yet'}
                    </span>
                </div>
                <div style="font-size: 13px; color: #64748b; margin-top: 4px;">
                    PSI distribution tests + KS-test + Chi-Square contingency.
                </div>
            </div>
            """, unsafe_allow_html=True)

            if engine.APP["quality_comparison"] is not None:
                st.markdown("### ⚖️ Quality Score Comparison")
                b_score = metrics['Quality Score (%)']
                a_score = appr_metrics['Quality Score (%)']
                delta = round(a_score - b_score, 2)
                st.metric(
                    label="Quality Score Improvement",
                    value=f"{a_score}%",
                    delta=f"{delta:+} percentage points"
                )

    else:
        # Empty State
        st.markdown("""
        <div class="empty-state">
            <div style="font-size: 48px; margin-bottom: 12px;">📁</div>
            <h3>No Dataset Loaded in Active Session</h3>
            <p>
                Upload your CSV, XLS, or XLSX file in the <b>Data Quality Analysis</b> tab,
                or load the built-in realistic demo dataset to immediately test the platform's
                imputation, anomaly detection, drift analysis, and reporting engines.
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            if st.button("⚡ Quick Load Demo Dataset (with Quality Defects)", type="primary", use_container_width=True):
                with st.spinner("Generating sample dataset with missing values, outliers, duplicates & text issues..."):
                    sample_df = generate_sample_quality_dataset()
                    summary, issues, orig, prop, appr = engine.analyze_dataset(sample_df)
                    st.session_state["dataset_name"] = "sample_quality_data.csv"
                    st.toast("Demo dataset analyzed successfully!", icon="✅")
                    st.rerun()

        st.markdown("---")
        st.markdown("### 🌟 Intelligent Architecture Capabilities")
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown("""
            **🌲 Random Forest Imputation**
            - Regresses missing numerical fields using predictive relationships.
            - Classifies categorical targets with balanced subsampling.
            - Fallbacks to mode/mean when samples are sparse.
            """)
        with f2:
            st.markdown("""
            **🧠 Dual-Engine Anomaly Detection**
            - **Autoencoder Neural Net:** Measures reconstruction loss.
            - **Isolation Forest:** Evaluates partition depths.
            - **Explainability:** Identifies top contributing feature loss.
            """)
        with f3:
            st.markdown("""
            **📈 Multi-Metric Drift Detection**
            - **PSI:** Population Stability Index with 10 quantiles.
            - **KS-Test:** Two-sample Kolmogorov-Smirnov for numerics.
            - **Chi-Square:** Contingency matrix test for categoricals.
            """)


# ================================================================
# PAGE 2: DATA QUALITY ANALYSIS
# ================================================================

elif page == "🔍 Data Quality Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Data Quality Analysis & Profiling</h1>
        <p>Comprehensive issue detection, statistical profiling, and Random Forest machine learning imputation.</p>
    </div>
    """, unsafe_allow_html=True)

    up_col1, up_col2 = st.columns([3, 1])

    with up_col1:
        uploaded_file = st.file_uploader(
            "Upload Dataset (CSV, XLS, or XLSX)",
            type=["csv", "xlsx", "xls"],
            help="Upload the raw dataset to profile for missing values, duplicates, and anomalies."
        )

    with up_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        load_sample = st.button("⚡ Load Demo Dataset", use_container_width=True)

    # Action trigger
    if load_sample:
        with st.spinner("Analyzing demo dataset..."):
            sample_df = generate_sample_quality_dataset()
            summary, issues, orig, prop, appr = engine.analyze_dataset(sample_df)
            st.session_state["dataset_name"] = "sample_quality_data.csv"
            st.session_state["analysis_summary"] = summary
            st.toast("Demo dataset loaded and analyzed!", icon="✅")

    if uploaded_file is not None:
        if st.button("🚀 Analyze Uploaded Dataset", type="primary"):
            with st.spinner("Analyzing dataset quality and running Random Forest imputation..."):
                try:
                    summary, issues, orig, prop, appr = engine.analyze_dataset(uploaded_file)
                    st.session_state["dataset_name"] = uploaded_file.name
                    st.session_state["analysis_summary"] = summary
                    st.toast("Analysis complete!", icon="✅")
                except Exception as exc:
                    st.error(f"Error analyzing dataset: {exc}")

    # Display results if dataset is loaded
    if engine.APP["original"] is not None:
        metrics = engine.calculate_quality_metrics(engine.APP["original"])

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

        # Dataset summary callout
        summary_text = st.session_state.get(
            "analysis_summary",
            f"Dataset loaded: {metrics['Rows']:,} rows, {metrics['Columns']:,} columns. "
            f"Initial Quality Score: {metrics['Quality Score (%)']}%."
        )
        st.markdown(f"""
        <div class="callout-box">
            <div style="font-weight: 700; color: #0d9488; margin-bottom: 6px;">📋 Dataset Profiling Summary</div>
            <div style="white-space: pre-line; font-size: 14px; color: #334155;">{summary_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # Metric cards row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows", f"{metrics['Rows']:,}")
        m2.metric("Columns", f"{metrics['Columns']:,}")
        m3.metric("Missing Cells", f"{metrics['Missing Cells']:,}")
        m4.metric("Duplicate Rows", f"{metrics['Duplicate Rows']:,}")
        m5.metric("Quality Score", f"{metrics['Quality Score (%)']}%")

        st.markdown("### 🚨 Detected Quality Issues")
        if engine.APP["issues"] is not None and not engine.APP["issues"].empty:
            st.dataframe(engine.APP["issues"], use_container_width=True)
        else:
            st.success("No issues detected.")

        if engine.APP["imputation_report"] is not None and not engine.APP["imputation_report"].empty:
            st.markdown("### 🌲 Imputation Plan & Methodology")
            st.dataframe(engine.APP["imputation_report"], use_container_width=True)

        st.markdown("### 🔎 Dataset Inspection & Previews")
        tab1, tab2, tab3 = st.tabs([
            "1. Original Dataset (Head 15)",
            "2. Proposed Dataset (Imputed Values)",
            "3. Current Approved Dataset"
        ])

        with tab1:
            st.caption("Raw uploaded data before any imputation or deduplication.")
            st.dataframe(engine.dataframe_preview(engine.APP["original"], 15), use_container_width=True)

        with tab2:
            st.caption("Data with Random Forest & mode/mean imputed values. (Pending human approval)")
            st.dataframe(engine.dataframe_preview(engine.APP["proposed"], 15), use_container_width=True)

        with tab3:
            st.caption("Active approved dataset. Matches original until you approve suggestions on Page 3.")
            st.dataframe(engine.dataframe_preview(engine.APP["approved"], 15), use_container_width=True)

        st.info("💡 **Next Step:** Head to **✍️ Human Approval & Override** in the sidebar to review, accept, or reject suggested imputations and duplicates.")

    else:
        st.info("Upload a dataset or click **⚡ Load Demo Dataset** above to start the analysis.")


# ================================================================
# PAGE 3: HUMAN APPROVAL AND DATA CORRECTION
# ================================================================

elif page == "✍️ Human Approval & Override":
    st.markdown("""
    <div class="main-header">
        <h1>✍️ Human-in-the-Loop Approval & Manual Override</h1>
        <p>Review suggested values, control deduplication, and apply manual cell overrides with full auditability.</p>
    </div>
    """, unsafe_allow_html=True)

    if engine.APP["original"] is None:
        st.warning("Please upload and analyze a dataset on the **Data Quality Analysis** page first.")
        if st.button("⚡ Or Load Demo Dataset Now"):
            sample_df = generate_sample_quality_dataset()
            engine.analyze_dataset(sample_df)
            st.session_state["dataset_name"] = "sample_quality_data.csv"
            st.rerun()
    else:
        st.markdown("""
        <div class="callout-box" style="border-left-color: #f59e0b;">
            <div style="font-weight: 700; color: #b45309; margin-bottom: 4px;">⚠️ Human Approval Governance</div>
            <div style="font-size: 13px; color: #475569;">
                Proposed machine learning imputation values are <b>NOT</b> automatically applied to the production dataset.
                You have full authority to accept or reject suggestions, choose how duplicates are handled, and perform manual cell overrides.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Section 1: Accept / Reject suggestions
        st.markdown("### 1. Imputation Decisions")
        col_acc, col_rej = st.columns(2)

        with col_acc:
            if st.button("✅ Accept Suggested Values", type="primary", use_container_width=True):
                msg, _ = engine.approve_suggestions()
                st.session_state["approval_message"] = msg
                st.toast(msg, icon="✅")
                st.rerun()

        with col_rej:
            if st.button("❌ Reject Suggested Values", use_container_width=True):
                msg, _ = engine.reject_suggestions()
                st.session_state["approval_message"] = msg
                st.toast(msg, icon="ℹ️")
                st.rerun()

        if st.session_state.get("approval_message"):
            st.info(f"**Status:** {st.session_state['approval_message']}")

        st.markdown("---")

        # Section 2: Duplicate Handling
        st.markdown("### 2. Duplicate Row Governance")
        col_dup_rem, col_dup_keep = st.columns(2)

        with col_dup_rem:
            if st.button("🗑️ Remove Duplicate Rows", use_container_width=True):
                msg, _ = engine.remove_duplicates()
                st.session_state["duplicate_message"] = msg
                st.toast(msg, icon="🗑️")
                st.rerun()

        with col_dup_keep:
            if st.button("🛡️ Keep Duplicate Rows", use_container_width=True):
                msg, _ = engine.keep_duplicates()
                st.session_state["duplicate_message"] = msg
                st.toast(msg, icon="🛡️")
                st.rerun()

        if st.session_state.get("duplicate_message"):
            st.info(f"**Status:** {st.session_state['duplicate_message']}")

        st.markdown("---")

        # Section 3: Manual Human Override
        st.markdown("### 3. Manual Cell Override")
        st.markdown("Specify a zero-based row index and column name to replace any specific cell with your manual value.")

        current_approved = engine.APP["approved"]
        col_names = list(current_approved.columns) if current_approved is not None else []
        max_rows = len(current_approved) if current_approved is not None else 1

        ov_col1, ov_col2, ov_col3 = st.columns([1, 2, 2])

        with ov_col1:
            row_idx = st.number_input(
                "Row Index (0-based)",
                min_value=0,
                max_value=max(0, max_rows - 1),
                value=0,
                step=1
            )

        with ov_col2:
            target_col = st.selectbox(
                "Column Name",
                options=col_names,
                index=0 if col_names else 0
            )

        # Dynamic display of current cell value
        current_val_preview = ""
        if current_approved is not None and target_col in current_approved.columns and 0 <= row_idx < len(current_approved):
            current_val_preview = str(current_approved.iloc[row_idx][target_col])

        with ov_col3:
            new_val = st.text_input(
                f"New Value (Current: '{current_val_preview}')",
                placeholder="Enter replacement value"
            )

        if st.button("✏️ Apply Human Override", type="primary"):
            if target_col:
                msg, _ = engine.apply_human_override(row_idx, target_col, new_val)
                st.session_state["override_message"] = msg
                st.toast(msg, icon="✏️")
                st.rerun()
            else:
                st.error("Please select a valid column.")

        if st.session_state.get("override_message"):
            st.info(f"**Override Status:** {st.session_state['override_message']}")

        st.markdown("---")

        # Section 4: Current Approved Dataset Preview
        st.markdown("### 📋 Approved Dataset Preview")
        if engine.APP["approved"] is not None:
            metrics_appr = engine.calculate_quality_metrics(engine.APP["approved"])
            st.caption(f"Currently approved dataset contains {metrics_appr['Rows']:,} rows and {metrics_appr['Columns']} columns. Missing cells: {metrics_appr['Missing Cells']:,}.")
            st.dataframe(engine.dataframe_preview(engine.APP["approved"], 15), use_container_width=True)

        st.markdown("---")

        # Section 5: Downloads
        st.markdown("### 📥 Download Results")
        d1, d2 = st.columns(2)

        with d1:
            if st.button("📊 Generate Quality Excel Report", use_container_width=True):
                with st.spinner("Generating multi-tab Excel report..."):
                    path = engine.download_quality_report()
                    data_bytes = get_file_bytes(path)
                    if data_bytes:
                        st.download_button(
                            label="⬇️ Download Quality Report (.xlsx)",
                            data=data_bytes,
                            file_name="data_quality_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

        with d2:
            if st.button("💾 Export Approved Dataset (.xlsx)", use_container_width=True):
                with st.spinner("Exporting approved dataset to Excel..."):
                    path = engine.download_approved_dataset()
                    data_bytes = get_file_bytes(path)
                    if data_bytes:
                        st.download_button(
                            label="⬇️ Download Approved Dataset (.xlsx)",
                            data=data_bytes,
                            file_name="final_approved_dataset.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )


# ================================================================
# PAGE 4: ANOMALY DETECTION
# ================================================================

elif page == "🧠 Anomaly Detection":
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Explainable Deep Learning Anomaly Detection</h1>
        <p>Dual consensus architecture: TensorFlow Autoencoder reconstruction loss + Scikit-Learn Isolation Forest.</p>
    </div>
    """, unsafe_allow_html=True)

    anom_source = st.radio(
        "Dataset Source",
        ["Use currently approved dataset from session", "Upload a separate dataset for anomaly detection"],
        horizontal=True
    )

    dataset_to_analyze = None

    if anom_source == "Use currently approved dataset from session":
        if engine.APP["approved"] is not None:
            dataset_to_analyze = engine.APP["approved"]
            st.caption(f"Using approved dataset with {len(dataset_to_analyze)} rows and {len(dataset_to_analyze.columns)} columns.")
        elif engine.APP["original"] is not None:
            dataset_to_analyze = engine.APP["original"]
            st.caption(f"Using original dataset with {len(dataset_to_analyze)} rows and {len(dataset_to_analyze.columns)} columns.")
        else:
            st.info("No active dataset in session. Upload a dataset below or load a demo.")

    if dataset_to_analyze is None:
        anom_file = st.file_uploader(
            "Upload Dataset for Anomaly Detection (CSV, XLS, XLSX)",
            type=["csv", "xlsx", "xls"]
        )
        if anom_file is not None:
            dataset_to_analyze = anom_file

    st.markdown("### ⚙️ Autoencoder Threshold Sensitivity")
    percentile = st.slider(
        "Autoencoder Error Threshold Percentile",
        min_value=80.0,
        max_value=99.9,
        value=95.0,
        step=0.5,
        help="Rows with reconstruction error exceeding this percentile are flagged as Autoencoder anomalies. Default is 95.0%."
    )

    if st.button("🚀 Run Anomaly Detection", type="primary"):
        if dataset_to_analyze is not None:
            with st.spinner("Training Autoencoder Neural Network (50 epochs) & Fitting Isolation Forest..."):
                try:
                    summary, result, plot_path = engine.run_anomaly_detection(dataset_to_analyze, percentile=percentile)
                    st.session_state["anomaly_summary"] = summary
                    st.toast("Anomaly detection completed successfully!", icon="✅")
                except Exception as exc:
                    st.error(f"Anomaly detection error: {exc}")
        else:
            st.error("Please provide or upload a dataset first.")

    # Display results if available
    if engine.APP["anomaly_report"] is not None:
        result = engine.APP["anomaly_report"]
        ae_count = int(result["Autoencoder_Anomaly"].sum())
        iso_count = int(result["IsolationForest_Anomaly"].sum())
        both_count = int(result["Both_Models_Flagged"].sum())
        either_count = int(result["Either_Model_Flagged"].sum())

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

        # 4 KPI Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Autoencoder Anomalies", f"{ae_count:,}")
        k2.metric("Isolation Forest Anomalies", f"{iso_count:,}")
        k3.metric("Flagged by Both Models", f"{both_count:,}")
        k4.metric("Flagged by Either Model", f"{either_count:,}")

        # Summary box
        if "anomaly_summary" in st.session_state:
            st.markdown(f"""
            <div class="callout-box">
                <div style="font-weight: 700; color: #0d9488; margin-bottom: 6px;">🧠 Anomaly Analysis Summary</div>
                <div style="white-space: pre-line; font-size: 14px; color: #334155;">{st.session_state['anomaly_summary']}</div>
            </div>
            """, unsafe_allow_html=True)

        col_plot, col_dl = st.columns([3, 1])
        with col_plot:
            st.markdown("### 📈 Autoencoder Reconstruction Error Distribution")
            if engine.APP["anomaly_plot"] and os.path.exists(engine.APP["anomaly_plot"]):
                st.image(engine.APP["anomaly_plot"], use_container_width=True)

        with col_dl:
            st.markdown("### 📥 Excel Report")
            st.write("Export detailed row-by-row anomaly classifications and feature loss contributions.")
            if st.button("📊 Download Anomaly Excel Report", use_container_width=True):
                path = engine.download_anomaly_report()
                data_bytes = get_file_bytes(path)
                if data_bytes:
                    st.download_button(
                        label="⬇️ Save Anomaly Report (.xlsx)",
                        data=data_bytes,
                        file_name="anomaly_detection_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

        st.markdown("### 🔍 Anomaly Results & Top Contributing Feature Explanations")
        filter_option = st.selectbox(
            "Filter Displayed Records",
            ["Flagged by Either Model (Recommended)", "Flagged by Both Models", "Autoencoder Only", "Isolation Forest Only", "All Rows"]
        )

        if filter_option == "Flagged by Either Model (Recommended)":
            display_df = result[result["Either_Model_Flagged"]]
        elif filter_option == "Flagged by Both Models":
            display_df = result[result["Both_Models_Flagged"]]
        elif filter_option == "Autoencoder Only":
            display_df = result[result["Autoencoder_Anomaly"]]
        elif filter_option == "Isolation Forest Only":
            display_df = result[result["IsolationForest_Anomaly"]]
        else:
            display_df = result

        # Display important columns first
        priority_cols = [
            "Autoencoder_Error", "Autoencoder_Anomaly",
            "IsolationForest_Score", "IsolationForest_Anomaly",
            "Both_Models_Flagged", "Either_Model_Flagged",
            "Top_Contributing_Features"
        ]
        other_cols = [c for c in display_df.columns if c not in priority_cols and not c.startswith("AE_Error_")]
        show_cols = priority_cols + other_cols

        st.caption(f"Showing {len(display_df):,} records.")
        st.dataframe(display_df[show_cols], use_container_width=True)

    else:
        st.info("Configure your threshold and click **Run Anomaly Detection** above.")


# ================================================================
# PAGE 5: DATA DRIFT ANALYSIS
# ================================================================

elif page == "📈 Data Drift Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Data Drift & Population Shift Analysis</h1>
        <p>Population Stability Index (PSI) with Kolmogorov-Smirnov and Chi-Square statistical validation.</p>
    </div>
    """, unsafe_allow_html=True)

    c_ref, c_cur = st.columns(2)

    with c_ref:
        ref_file = st.file_uploader(
            "Upload Reference Dataset (Baseline)",
            type=["csv", "xlsx", "xls"],
            key="ref_file"
        )

    with c_cur:
        cur_file = st.file_uploader(
            "Upload Current Dataset (Production / Recent)",
            type=["csv", "xlsx", "xls"],
            key="cur_file"
        )

    col_btn1, col_btn2 = st.columns([2, 1])

    with col_btn2:
        if st.button("⚡ Load Demo Drift Datasets", use_container_width=True):
            ref_df, cur_df = generate_sample_drift_datasets()
            st.session_state["demo_ref"] = ref_df
            st.session_state["demo_cur"] = cur_df
            st.toast("Demo reference and current datasets loaded!", icon="✅")

    ready_ref = ref_file if ref_file is not None else st.session_state.get("demo_ref")
    ready_cur = cur_file if cur_file is not None else st.session_state.get("demo_cur")

    with col_btn1:
        if st.button("🚀 Run Multiple Drift Analysis", type="primary", use_container_width=True):
            if ready_ref is not None and ready_cur is not None:
                with st.spinner("Calculating PSI, Kolmogorov-Smirnov distributions & Chi-Square statistics..."):
                    try:
                        summary, drift_report, plot_path = engine.run_drift_analysis(ready_ref, ready_cur)
                        st.session_state["drift_summary"] = summary
                        st.toast("Data drift analysis completed!", icon="✅")
                    except Exception as exc:
                        st.error(f"Error during drift analysis: {exc}")
            else:
                st.error("Please upload or load both Reference and Current datasets.")

    # Display results if available
    if engine.APP["drift_report"] is not None:
        drift_df = engine.APP["drift_report"]
        high_cnt = int((drift_df["Drift Level"] == "High").sum())
        mod_cnt = int((drift_df["Drift Level"] == "Moderate").sum())
        low_cnt = int((drift_df["Drift Level"] == "Low").sum())
        sig_cnt = int((drift_df["Statistical Result"] == "Statistically significant").sum())

        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Low Drift Columns", f"{low_cnt}")
        k2.metric("Moderate Drift (PSI ≥ 0.10)", f"{mod_cnt}")
        k3.metric("High Drift (PSI ≥ 0.25)", f"{high_cnt}")
        k4.metric("Statistically Significant (p < 0.05)", f"{sig_cnt}")

        if "drift_summary" in st.session_state:
            st.markdown(f"""
            <div class="callout-box">
                <div style="font-weight: 700; color: #0d9488; margin-bottom: 6px;">📈 Data Drift Summary</div>
                <div style="white-space: pre-line; font-size: 14px; color: #334155;">{st.session_state['drift_summary']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Population Stability Index (PSI) Visualization")
        if engine.APP["drift_plot"] and os.path.exists(engine.APP["drift_plot"]):
            st.image(engine.APP["drift_plot"], use_container_width=True)

        st.markdown("### 📋 Statistical Drift Analysis Table")
        st.dataframe(drift_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📥 Download Drift Report")
        if st.button("📊 Generate Drift Excel Report"):
            path = engine.download_drift_report()
            data_bytes = get_file_bytes(path)
            if data_bytes:
                st.download_button(
                    label="⬇️ Download Drift Report (.xlsx)",
                    data=data_bytes,
                    file_name="data_drift_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    else:
        st.info("Upload reference & current datasets or click **⚡ Load Demo Drift Datasets** above.")


# ================================================================
# PAGE 6: QUALITY COMPARISON
# ================================================================

elif page == "⚖️ Quality Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Before-and-After Quality Comparison</h1>
        <p>Quantify data quality improvements before and after human approval, imputation, and deduplication.</p>
    </div>
    """, unsafe_allow_html=True)

    if engine.APP["original"] is None or engine.APP["approved"] is None:
        st.warning("Please upload and analyze a dataset first.")
        if st.button("⚡ Load Demo Dataset"):
            sample_df = generate_sample_quality_dataset()
            engine.analyze_dataset(sample_df)
            engine.approve_suggestions()
            engine.remove_duplicates()
            st.session_state["dataset_name"] = "sample_quality_data.csv"
            st.rerun()
    else:
        if st.button("🚀 Compare Original vs Approved Dataset", type="primary"):
            with st.spinner("Computing quality delta metrics and generating comparison charts..."):
                summary, comp_df, plot_path = engine.compare_quality_before_after()
                st.session_state["comparison_summary"] = summary
                st.toast("Comparison computed!", icon="✅")

        if engine.APP["quality_comparison"] is not None:
            comp_df = engine.APP["quality_comparison"]

            # Extract key delta metrics
            def get_metric_val(metric_name, col):
                row = comp_df[comp_df["Metric"] == metric_name]
                if not row.empty:
                    return row.iloc[0][col]
                return 0

            b_score = get_metric_val("Quality Score (%)", "Before")
            a_score = get_metric_val("Quality Score (%)", "After")
            score_delta = round(float(a_score) - float(b_score), 2)

            b_miss = get_metric_val("Missing Cells", "Before")
            a_miss = get_metric_val("Missing Cells", "After")
            miss_delta = int(a_miss) - int(b_miss)

            b_dup = get_metric_val("Duplicate Rows", "Before")
            a_dup = get_metric_val("Duplicate Rows", "After")
            dup_delta = int(a_dup) - int(b_dup)

            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Overall Quality Score", f"{a_score}%", delta=f"{score_delta:+} pp")
            c2.metric("Missing Cells", f"{a_miss:,}", delta=f"{miss_delta:,} cells")
            c3.metric("Duplicate Rows", f"{a_dup:,}", delta=f"{dup_delta:,} rows")

            if "comparison_summary" in st.session_state:
                st.markdown(f"""
                <div class="callout-box">
                    <div style="font-weight: 700; color: #0d9488; margin-bottom: 6px;">⚖️ Comparison Summary</div>
                    <div style="white-space: pre-line; font-size: 14px; color: #334155;">{st.session_state['comparison_summary']}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 📊 Before vs After Comparison Chart")
            if engine.APP["quality_comparison_plot"] and os.path.exists(engine.APP["quality_comparison_plot"]):
                st.image(engine.APP["quality_comparison_plot"], use_container_width=True)

            st.markdown("### 📋 Metric Change Audit Table")
            st.dataframe(comp_df, use_container_width=True)

        else:
            st.info("Click **🚀 Compare Original vs Approved Dataset** above to generate the before-and-after audit.")


# ================================================================
# PAGE 7: COMBINED REPORTS
# ================================================================

elif page == "📑 Combined Reports":
    st.markdown("""
    <div class="main-header">
        <h1>📑 Combined Monitoring & Audit Reports</h1>
        <p>Export an all-in-one multi-sheet Excel workbook containing all session analyses, models, and datasets.</p>
    </div>
    """, unsafe_allow_html=True)

    if engine.APP["original"] is None:
        st.warning("Please upload and analyze a dataset first to generate reports.")
        if st.button("⚡ Load Demo Dataset"):
            sample_df = generate_sample_quality_dataset()
            engine.analyze_dataset(sample_df)
            st.session_state["dataset_name"] = "sample_quality_data.csv"
            st.rerun()
    else:
        st.markdown("### 📦 Available Worksheets in Report")

        # Checklist of available modules
        sheets_status = [
            ("Original Dataset", engine.APP["original"] is not None),
            ("Proposed Dataset (Imputed)", engine.APP["proposed"] is not None),
            ("Approved Dataset", engine.APP["approved"] is not None),
            ("Detected Quality Issues", engine.APP["issues"] is not None),
            ("Initial Quality Summary", engine.APP["quality_summary"] is not None),
            ("Imputation Plan & Report", engine.APP["imputation_report"] is not None),
            ("Quality Comparison (Before vs After)", engine.APP["quality_comparison"] is not None),
            ("Anomaly Results & Model Summary", engine.APP["anomaly_report"] is not None),
            ("Data Drift Results & Datasets", engine.APP["drift_report"] is not None),
        ]

        cols = st.columns(3)
        for i, (sheet_title, ready) in enumerate(sheets_status):
            with cols[i % 3]:
                if ready:
                    st.markdown(f"✅ **{sheet_title}** *(Ready)*")
                else:
                    st.markdown(f"⚪ *{sheet_title}* *(Not run)*")

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

        if st.button("📊 Generate Combined Excel Report", type="primary", use_container_width=True):
            with st.spinner("Generating consolidated Excel workbook..."):
                try:
                    path = engine.download_combined_report()
                    data_bytes = get_file_bytes(path)
                    if data_bytes:
                        st.success("✅ Combined report generated successfully!")
                        st.download_button(
                            label="⬇️ Download Combined Monitoring Report (.xlsx)",
                            data=data_bytes,
                            file_name="combined_monitoring_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                except Exception as exc:
                    st.error(f"Error generating combined report: {exc}")
