"""
Streamlit Web Application: Healthcare Cost Pattern Discovery & Anomaly Detection
Clean, professional UI with custom HTML styling and minimal emojis.
"""

import os
import joblib
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Healthcare Cost Clustering & Anomaly Detection",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean CSS Design System
st.markdown("""
<style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0f2b48 0%, #1a365d 60%, #2b6cb0 100%);
        color: #ffffff;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(15, 43, 72, 0.12);
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 0.96rem;
        color: #cbd5e0;
        margin-top: 6px;
        font-weight: 400;
    }

    /* Cards */
    .ui-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .ui-card-header {
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #718096;
        margin-bottom: 8px;
    }
    .ui-card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 10px;
    }

    /* Status Badges with CSS dots */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-normal {
        background-color: #def7ec;
        color: #03543f;
        border: 1px solid #bcf0da;
    }
    .status-alert {
        background-color: #fde8e8;
        color: #9b1c1c;
        border: 1px solid #fbd5d5;
    }
    .dot-normal {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #0e9f6e;
        display: inline-block;
        margin-right: 8px;
    }
    .dot-alert {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #e02424;
        display: inline-block;
        margin-right: 8px;
    }

    /* Stat Highlight Boxes */
    .stat-box {
        background: #f8fafc;
        border: 1px solid #edf2f7;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2b6cb0;
        margin: 4px 0;
    }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* Cohort Mini Cards */
    .cohort-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        border-top: 3px solid #3182ce;
        text-align: center;
    }
    .cohort-name {
        font-size: 0.85rem;
        font-weight: 600;
        color: #2d3748;
        min-height: 38px;
    }
    .cohort-cost {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1a365d;
        margin: 6px 0;
    }
    .cohort-meta {
        font-size: 0.78rem;
        color: #718096;
    }

    /* Details List */
    .details-row {
        display: flex;
        justify-content: space-between;
        padding: 7px 0;
        border-bottom: 1px solid #edf2f7;
        font-size: 0.88rem;
    }
    .details-row:last-child {
        border-bottom: none;
    }
    .details-label {
        color: #718096;
    }
    .details-value {
        font-weight: 600;
        color: #2d3748;
    }
</style>
""", unsafe_allow_html=True)


# Load single model artifact
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "model.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "model.joblib")
    return joblib.load(model_path)

model_data = load_model()
df_data = model_data['df']
profiles = model_data['cluster_profiles']
metrics_table = model_data['metrics_table']

# Check FastAPI status
API_URL = "http://127.0.0.1:8000"
def is_api_running():
    try:
        r = requests.get(f"{API_URL}/", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False

api_active = is_api_running()


# Header Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Healthcare Cost Pattern Discovery & Anomaly Detection</div>
    <div class="hero-subtitle">Unsupervised policyholder segmentation with K-Means and unusual billing detection with DBSCAN</div>
</div>
""", unsafe_allow_html=True)


# Sidebar Inputs
st.sidebar.markdown("### Policyholder Input")

age = st.sidebar.slider("Age (Years)", 18, 70, 32)
bmi = st.sidebar.slider("Body Mass Index (BMI)", 15.0, 55.0, 28.0, 0.1)
smoker = st.sidebar.selectbox("Smoking Status", ["No", "Yes"])
charges = st.sidebar.number_input("Annual Healthcare Charges (USD)", 500.0, 65000.0, 4500.0, 250.0)

st.sidebar.markdown("---")
if api_active:
    st.sidebar.markdown("<small style='color:#0e9f6e; font-weight:600;'>Service Status: FastAPI Backend Connected</small>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<small style='color:#3182ce; font-weight:600;'>Service Status: Standalone Engine Active</small>", unsafe_allow_html=True)


# Prediction Logic
smoker_enc = 1 if smoker == "Yes" else 0
raw_input = np.array([[age, bmi, smoker_enc, charges]])
scaled_input = model_data['scaler'].transform(raw_input)

# K-Means prediction
pred_cluster = int(model_data['kmeans'].predict(scaled_input)[0])
pred_profile = next((p for p in profiles if p['cluster_id'] == pred_cluster), profiles[0])

# DBSCAN Outlier check (distance to nearest core point > eps)
core_samples = model_data['dbscan_core_samples']
eps_val = float(model_data['dbscan_eps'])
min_dist = float(np.min(np.linalg.norm(core_samples - scaled_input, axis=1)))
is_unusual = bool(min_dist > eps_val)


# Main Tabs (Clean labels without emoji clutter)
tab1, tab2, tab3 = st.tabs(["Policyholder Assessment", "Clustering Metrics", "Domain Visualizations"])


# ----------------- TAB 1: Policyholder Assessment -----------------
with tab1:
    st.markdown("#### Input Evaluation")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="ui-card">
            <div class="ui-card-header">K-Means Segment Assignment</div>
            <div class="ui-card-title">{pred_profile['persona_name']}</div>
            <div style="margin-bottom: 14px;">
                <span class="status-badge status-normal" style="font-size:0.78rem;">
                    Expenditure Tier: {pred_profile['cost_tier']}
                </span>
                <span style="font-size:0.8rem; color:#718096; margin-left: 8px;">
                    Cluster ID: {pred_cluster}
                </span>
            </div>
            <div class="details-row">
                <span class="details-label">Cohort Average Charges:</span>
                <span class="details-value">${pred_profile['avg_charges']:,.2f}</span>
            </div>
            <div class="details-row">
                <span class="details-label">Cohort Median Charges:</span>
                <span class="details-value">${pred_profile['median_charges']:,.2f}</span>
            </div>
            <div class="details-row">
                <span class="details-label">Cohort Average Age:</span>
                <span class="details-value">{pred_profile['avg_age']} years</span>
            </div>
            <div class="details-row">
                <span class="details-label">Cohort Average BMI:</span>
                <span class="details-value">{pred_profile['avg_bmi']} kg/m²</span>
            </div>
            <div class="details-row">
                <span class="details-label">Smoker Rate in Group:</span>
                <span class="details-value">{pred_profile['smoker_pct']}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if is_unusual:
            verdict_badge = '<span class="status-badge status-alert"><span class="dot-alert"></span>Atypical Record (DBSCAN Outlier)</span>'
            verdict_text = f"The Euclidean distance to the nearest dense cluster core is <b>{min_dist:.2f}</b>, which exceeds the density boundary (eps = <b>{eps_val:.2f}</b>). This case falls into the noise region and is flagged for review."
        else:
            verdict_badge = '<span class="status-badge status-normal"><span class="dot-normal"></span>Typical Billing Pattern</span>'
            verdict_text = f"The Euclidean distance to the nearest dense cluster core is <b>{min_dist:.2f}</b>, which is well within the cluster density boundary (eps = <b>{eps_val:.2f}</b>)."

        st.markdown(f"""
        <div class="ui-card">
            <div class="ui-card-header">DBSCAN Anomaly Detection</div>
            <div style="margin: 6px 0 14px 0;">{verdict_badge}</div>
            <p style="font-size: 0.9rem; color: #4a5568; line-height: 1.5;">{verdict_text}</p>
            <hr style="border: none; border-top: 1px solid #edf2f7; margin: 16px 0;">
            <div class="details-row">
                <span class="details-label">Distance to Nearest Core Point:</span>
                <span class="details-value">{min_dist:.3f}</span>
            </div>
            <div class="details-row">
                <span class="details-label">DBSCAN Epsilon Radius (eps):</span>
                <span class="details-value">{eps_val:.2f}</span>
            </div>
            <div class="details-row">
                <span class="details-label">Dataset Outliers Identified:</span>
                <span class="details-value">63 records (4.71%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### All 4 Discovered Healthcare Cohorts")
    p_cols = st.columns(4)
    border_colors = ["#3182ce", "#e53e3e", "#38a169", "#d69e2e"]
    for i, p in enumerate(profiles):
        with p_cols[i]:
            st.markdown(f"""
            <div class="cohort-card" style="border-top-color: {border_colors[i]};">
                <div class="cohort-name">{p['persona_name']}</div>
                <div class="cohort-cost">${p['avg_charges']:,.0f}</div>
                <div class="cohort-meta">{p['count']} members ({p['share_pct']}%)</div>
                <div class="cohort-meta" style="margin-top: 4px;">Smokers: <b>{p['smoker_pct']}%</b> | Avg Age: <b>{p['avg_age']}</b></div>
            </div>
            """, unsafe_allow_html=True)


# ----------------- TAB 2: Clustering Metrics -----------------
with tab2:
    st.markdown("#### Three Standard Clustering Evaluation Metrics")
    st.caption("Unsupervised clustering models rely on geometric cluster quality metrics rather than classification labels.")

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-label">Metric 1: Silhouette Score</div>
            <div class="stat-number">-1 to +1</div>
            <small style="color: #718096;"><b>Higher is Better</b><br>Evaluates cohesion within each cluster vs. separation from neighbors.</small>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-label">Metric 2: Davies–Bouldin Index</div>
            <div class="stat-number">0 to &infin;</div>
            <small style="color: #718096;"><b>Lower is Better</b><br>Measures cluster overlap and scatter. 0 is ideal separation.</small>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown("""
        <div class="stat-box">
            <div class="stat-label">Metric 3: Noise / Outlier Rate</div>
            <div class="stat-number">4.71%</div>
            <small style="color: #718096;"><b>Target: 2% to 6%</b><br>Proportion of atypical records isolated as noise by DBSCAN.</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Side-by-Side Algorithm Comparison")
    
    # Styled table display
    m_df = pd.DataFrame(metrics_table)
    st.dataframe(m_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("##### K-Means Diagnostics: Elbow Curve & Silhouette Scan")
    scan_df = pd.DataFrame(model_data['kmeans_scan'])

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=scan_df['k'], y=scan_df['inertia'],
        mode='lines+markers', name='WCSS / Inertia (Elbow)',
        line=dict(color='#2b6cb0', width=3),
        marker=dict(size=8, color='#1a365d')
    ))
    fig_elbow.update_layout(
        title="WCSS vs. Number of Clusters (Elbow bend observed at K=4)",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (Within-Cluster Sum of Squares)",
        height=380,
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig_elbow, use_container_width=True)


# ----------------- TAB 3: Visualizations -----------------
with tab3:
    st.markdown("#### Domain Bivariate Visualizations (Real Features — No PCA)")
    st.caption("Visualizing on actual healthcare features ensures direct clinical and financial interpretability.")

    plot_df = df_data.copy()
    p_names = {p['cluster_id']: p['persona_name'] for p in profiles}
    plot_df['Group'] = plot_df['kmeans_cluster'].map(p_names)
    plot_df['DBSCAN_Pattern'] = plot_df['dbscan_cluster'].apply(
        lambda x: "Atypical Case (Outlier)" if x == -1 else "Dense Cluster"
    )

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### Charges vs. Age (K-Means Segments)")
        fig1 = px.scatter(
            plot_df, x='age', y='charges', color='Group',
            hover_data=['bmi', 'smoker'],
            labels={'age': 'Age (Years)', 'charges': 'Annual Charges (USD)'},
            template='plotly_white', height=460,
            color_discrete_sequence=['#2b6cb0', '#e53e3e', '#38a169', '#dd6b20']
        )
        fig1.add_trace(go.Scatter(
            x=[age], y=[charges], mode='markers',
            marker=dict(size=18, color='#d69e2e', symbol='star', line=dict(color='#000000', width=2)),
            name='Current Input'
        ))
        fig1.update_layout(margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig1, use_container_width=True)

    with g_col2:
        st.markdown("##### Charges vs. Age (DBSCAN Anomaly Map)")
        fig2 = px.scatter(
            plot_df, x='age', y='charges', color='DBSCAN_Pattern',
            color_discrete_map={"Atypical Case (Outlier)": "#e53e3e", "Dense Cluster": "#718096"},
            labels={'age': 'Age (Years)', 'charges': 'Annual Charges (USD)'},
            template='plotly_white', height=460
        )
        fig2.add_trace(go.Scatter(
            x=[age], y=[charges], mode='markers',
            marker=dict(size=18, color='#d69e2e', symbol='star', line=dict(color='#000000', width=2)),
            name='Current Input'
        ))
        fig2.update_layout(margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("##### Charges vs. BMI (Lifestyle Risk Boundary)")
    fig3 = px.scatter(
        plot_df, x='bmi', y='charges', color='Group', symbol='smoker',
        labels={'bmi': 'Body Mass Index (BMI)', 'charges': 'Annual Charges (USD)'},
        template='plotly_white', height=430,
        color_discrete_sequence=['#2b6cb0', '#e53e3e', '#38a169', '#dd6b20']
    )
    fig3.add_vline(x=30.0, line_dash="dash", line_color="#a0aec0", annotation_text="Obesity Line (BMI=30)")
    fig3.update_layout(margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig3, use_container_width=True)
