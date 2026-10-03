import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import time
import os

# 1. Page Config
st.set_page_config(
    page_title="Smart Traffic AI Hub",
    page_icon="🚥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚥 AI-Powered Smart Traffic Management & Safety Hub")
st.caption("Real-Time Autonomous Signal Control & Predictive Analytics Dashboard")

# 2. Load Dataset
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data', 'traffic_data.csv')
    return pd.read_csv(data_path)

df = load_data()

# 3. Sidebar Controls & Automation Toggle
st.sidebar.header("🎛️ Simulation Controls")

auto_mode = st.sidebar.checkbox("⚡ Enable Live Traffic Auto-Simulation", value=False)

if auto_mode:
    st.sidebar.info("Auto-simulation active: Generating dynamic real-time traffic feed.")
    vehicle_density = np.random.randint(10, 115)
    visibility = np.round(np.random.uniform(1.0, 10.0), 1)
    hour = np.random.randint(0, 24)
    wind_speed = np.round(np.random.uniform(2.0, 25.0), 1)
else:
    vehicle_density = st.sidebar.slider("Vehicle Density (vehicles/lane)", 5, 120, 65)
    visibility = st.sidebar.slider("Visibility (miles)", 0.5, 10.0, 5.0)
    hour = st.sidebar.slider("Hour of Day (24-Hour)", 0, 23, 18)
    wind_speed = st.sidebar.slider("Wind Speed (mph)", 0.0, 30.0, 12.0)

# Risk Calculation Logic
raw_risk = (vehicle_density * 0.03) + ((10 - visibility) * 0.4) + (wind_speed * 0.1) + (2.5 if (7 <= hour <= 10 or 17 <= hour <= 20) else 0)

if raw_risk < 3.8:
    risk_status = "LOW RISK"
    color_code = "#28a745"
    signal_action = "🟢 Normal Timing (30s Green)"
    alert_msg = "Traffic flow is optimal. No congestion detected."
elif raw_risk < 6.0:
    risk_status = "MEDIUM RISK"
    color_code = "#ffc107"
    signal_action = "🟡 Adaptive Extension (+15s Green)"
    alert_msg = "Moderate vehicle density. System adjusting signal timings."
else:
    risk_status = "HIGH RISK"
    color_code = "#dc3545"
    signal_action = "🔴 Max Extension (+30s Green) + Early Hazard Alert"
    alert_msg = "Critical congestion / hazard warning! Emergency signal sequence activated."

# 4. Interactive KPI Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(label="Vehicle Density", value=f"{vehicle_density} / lane", delta="Live Count")

with kpi2:
    st.markdown(f"**Accident Risk Status**")
    st.markdown(f"<h3 style='color: {color_code}; margin-top:-10px;'>{risk_status}</h3>", unsafe_allow_html=True)

with kpi3:
    st.metric(label="Visibility", value=f"{visibility} mi", delta="Weather Feed")

with kpi4:
    st.metric(label="Signal Strategy", value=signal_action)

st.warning(f"**System Status Message:** {alert_msg}")
st.markdown("---")

# 5. Visual Gauges & Live Analytics Section
tab1, tab2 = st.tabs(["📊 Interactive Gauges & Trends", "🗺️ Live Traffic Map"])

with tab1:
    col_gauge, col_trend = st.columns([1, 1])
    
    with col_gauge:
        st.subheader("Risk Severity Index Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(raw_risk * 10, 100),
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color_code},
                'steps': [
                    {'range': [0, 38], 'color': '#d4edda'},
                    {'range': [38, 60], 'color': '#fff3cd'},
                    {'range': [60, 100], 'color': '#f8d7da'}
                ],
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_trend:
        st.subheader("Hourly Peak Congestion Model")
        hourly_df = df.groupby("Hour")["Vehicle_Density"].mean().reset_index()
        fig_line = px.area(hourly_df, x="Hour", y="Vehicle_Density", title="24-Hour Traffic Pattern", color_discrete_sequence=['#1f77b4'])
        fig_line.add_vline(x=hour, line_dash="dash", line_color="red", annotation_text="Current Hour")
        fig_line.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    st.subheader("Geographic High-Risk Accident Hotspots")
    # Interactive Map with Folium
    m = folium.Map(location=[17.3850, 78.4867], zoom_start=12, tiles="OpenStreetMap") # Default to Hyderabad area coordinates
    
    # Sample Hotspot Markers
    hotspots = [
        {"loc": [17.4401, 78.3489], "name": "Intersection A (Gachibowli)", "risk": "High Risk 🔴"},
        {"loc": [17.4359, 78.4482], "name": "Junction B (Punjagutta)", "risk": "Medium Risk 🟡"},
        {"loc": [17.3616, 78.4747], "name": "Corridor C (Charminar)", "risk": "Low Risk 🟢"},
    ]
    
    for spot in hotspots:
        folium.Marker(
            location=spot["loc"],
            popup=f"<b>{spot['name']}</b><br>Status: {spot['risk']}",
            tooltip=spot["name"],
            icon=folium.Icon(color="red" if "High" in spot["risk"] else "orange" if "Medium" in spot["risk"] else "green")
        ).add_to(m)
        
    st_folium(m, width=1100, height=350)

# Auto-refresh loop for live mode
if auto_mode:
    time.sleep(2)
    st.rerun()