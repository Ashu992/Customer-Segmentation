import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Insight Dashboard",
    page_icon="👤",
    layout="wide"
)

# --- 2. PREMIUM LIGHT UI CUSTOMIZATION (CSS) ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E9ECEF; }
    
    /* Action Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #FF8C00, #FFA500);
        color: white !important;
        border: none;
        padding: 0.8rem 2rem;
        font-weight: 700;
        border-radius: 10px;
        width: 100%;
        transition: 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 140, 0, 0.3);
    }

    /* Persona & History Cards */
    .persona-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'audit_log' not in st.session_state:
    st.session_state.audit_log = []

# --- 4. ASSET LOADING ---
@st.cache_resource
def load_analytics_assets():
    try:
        # Loading objects as exported in  [Jupyter Notebook](http://localhost:8888/notebooks/data%20science%20self/Projects/Customer%20Segmentation/Notebook/Customer%20Segmentation.ipynb)
        scaler = joblib.load("models/scaler.pkl")
        pca = joblib.load("models/pca.pkl")
        gmm = joblib.load("models/gmm_model.pkl")
        return scaler, pca, gmm
    except Exception as e:
        st.error(f"⚠️ Required data files not found.")
        return None, None, None

scaler, pca, gmm = load_analytics_assets()

# --- 5. SIDEBAR: CONTROL CONSOLE ---
with st.sidebar:
    st.title("⚙️ Control Console")
    st.caption("Adjust parameters to simulate customer behavior.")
    st.divider()
    
    income = st.number_input("Annual Income ($)", 0, 500000, 55000)
    is_parent = st.radio("Parental Status", [0, 1], format_func=lambda x: "Non-Parent" if x==0 else "Parent")
    
    st.markdown("### 🛒 Spending Vectors")
    wines = st.slider("Wines", 0, 2000, 300)
    meat = st.slider("Meat Products", 0, 2000, 200)
    others = st.slider("Other Products", 0, 1000, 100)
    total_spent = wines + meat + others

    st.markdown("---")
    st.subheader("📋 Behavioral Audit Trail")
    if st.session_state.audit_log:
        log_df = pd.DataFrame(st.session_state.audit_log)
        st.dataframe(log_df[['Persona', 'Income']].tail(3), use_container_width=True)
        st.download_button("Export History", log_df.to_csv(index=False), "customer_insights.csv", "text/csv")

# --- 6. DATA PIPELINE ---
if scaler and pca and gmm:
    input_df = pd.DataFrame([[income, total_spent, wines, meat, is_parent]], 
                             columns=['Income', 'Spent', 'Wines', 'Meat', 'Is_Parent'])

    input_log = input_df.copy()
    for col in ['Income', 'Spent', 'Wines', 'Meat']:
        input_log[col] = np.log1p(input_df[col])

    scaled_x = scaler.transform(input_log)
    pca_x = pca.transform(scaled_x)

    # --- 7. MAIN WORKSPACE (TWO COLUMN) ---
    st.title("👤 Customer Insight Dashboard")
    st.markdown(f"Developed by **Ashutosh Tripathi**")
    st.divider()

    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        st.subheader("🎯 Persona Classification")
        if st.button("RUN ANALYSIS"):
            cluster_id = gmm.predict(pca_x)[0]
            
            personas = {
                0: {"name": "High-Value Elite", "icon": "🥂", "color": "#28A745", "strat": "Priority VIP loyalty and exclusive wine previews."},
                1: {"name": "Value-Conscious Family", "icon": "🛒", "color": "#007BFF", "strat": "Bulk-buy discounts and essential pantry bundles."},
                2: {"name": "Established Household", "icon": "🏠", "color": "#FD7E14", "strat": "Convenience-focused subscriptions and family meal kits."},
                3: {"name": "Emerging Minimalist", "icon": "🌱", "color": "#6F42C1", "strat": "First-purchase incentives and social media flash sales."}
            }
            
            p = personas.get(cluster_id, {"name": "Standard Customer", "icon": "👤", "color": "#6C757D", "strat": "General engagement."})
            
            st.session_state.audit_log.append({
                "Timestamp": datetime.now().strftime("%H:%M"),
                "Persona": p['name'], "Income": income, "Spent": total_spent
            })
            
            st.markdown(f"""
                <div class="persona-card">
                    <h1 style="color:{p['color']}; margin-top:0;">{p['icon']} {p['name']}</h1>
                    <p style="font-size:1.1rem; color: #495057;">Classification: <b>Group {cluster_id}</b></p>
                    <hr style="opacity:0.1;">
                    <h4 style="color:#FF8C00;">💡 Recommended Strategy:</h4>
                    <p style="color: #495057; font-weight:500;">{p['strat']}</p>
                </div>
            """, unsafe_allow_html=True)

    with right_col:
        st.subheader("📊 DNA Feature Attribution")
        
        categories = ['Income', 'Total Spend', 'Wines', 'Meat', 'Parent Status']
        values = [min(100, income/1000), min(100, total_spent/20), min(100, wines/10), min(100, meat/10), is_parent*100]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
              r=values + [values[0]],
              theta=categories + [categories[0]],
              fill='toself',
              line_color='#FF8C00',
              fillcolor='rgba(255, 140, 0, 0.1)'
        ))

        fig.update_layout(
          polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#DEE2E6")),
          showlegend=False,
          paper_bgcolor="rgba(0,0,0,0)",
          margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 8. HISTORICAL TREND ANALYSIS ---
    if len(st.session_state.audit_log) > 1:
        st.divider()
        st.subheader("📈 Longitudinal Session Analysis")
        hist_df = pd.DataFrame(st.session_state.audit_log)
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_df.index, 
            y=hist_df['Spent'], 
            mode='lines+markers', 
            name='Total Spend',
            line=dict(color='#FF8C00', width=3)
        ))
        fig_hist.update_layout(
            title="Spending Variance Across Sessions",
            xaxis_title="Sequence",
            yaxis_title="USD ($)",
            plot_bgcolor='white'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# --- FOOTER ---
st.markdown(f"<div style='text-align: center; color: #ADB5BD; font-size: 0.8rem; margin-top: 50px;'>Analytical Framework by <b>Ashutosh Tripathi</b> | © {datetime.now().year}</div>", unsafe_allow_html=True)