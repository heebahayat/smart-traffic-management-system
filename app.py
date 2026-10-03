import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import joblib
import os

st.set_page_config(page_title="Smart Traffic AI Hub", page_icon="🚥", layout="wide")

st.title("🚥 AI-Powered Smart Traffic Management System")
st.caption("Real-Time Autonomous Signal Control & Machine Learning Risk Analytics")

# Load Trained ML Model
@st.cache_resource
def load_ml_model():
    model_path = os.path.join('models', 'xgb_traffic_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_ml_model()

@st.cache_data
def load_data():
    return pd.read_csv('data/traffic_data.csv')

df = load_data()

# Sidebar Input Controls
st.sidebar.header("🎛️ Live Parameter Simulation")
vehicle_density = st.sidebar.slider("Vehicle Density (vehicles/lane)", 5, 120, 65)
visibility = st.sidebar.slider("Visibility (miles)", 0.5, 10.0, 5.0)
hour = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 18)
wind_speed = st.sidebar.slider("Wind Speed (mph)", 0.0, 30.0, 12.0)
temp = st.sidebar.slider("Temperature (°F)", 50.0, 100.0, 75.0)
humidity = st.sidebar.slider("Humidity (%)", 20.0, 100.0, 60.0)

# Build feature array for model prediction
input_data = pd.DataFrame([{
    'Hour': hour,
    'DayOfWeek': 3,
    'Temperature_F': temp,
    'Humidity_Pct': humidity,
    'Visibility_mi': visibility,
    'Wind_Speed_mph': wind_speed,
    'Vehicle_Density': vehicle_density,
    'Traffic_Signal': 1,
    'Crossing': 0
}])

# Make ML Prediction
if model is not None:
    predicted_class = model.predict(input_data)[0]
    risk_probs = model.predict_proba(input_data)[0]
    risk_score = risk_probs[predicted_class] * 100
else:
    predicted_class = 1
    risk_score = 50.0

# Map prediction to risk levels & signal actions
if predicted_class == 0:
    risk_status, color_code, signal_action = "LOW RISK", "#28a745", "🟢 Normal Signal Timing (30s Green)"
elif predicted_class == 1:
    risk_status, color_code, signal_action = "MEDIUM RISK", "#ffc107", "🟡 Extended Green Light (+15s)"
else:
    risk_status, color_code, signal_action = "HIGH RISK", "#dc3545", "🔴 Max Extension (+30s) + Caution Alert"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Vehicle Density", f"{vehicle_density} / lane")
col2.markdown(f"**ML Predicted Risk**<h3 style='color:{color_code}; margin:0;'>{risk_status}</h3>", unsafe_allow_html=True)
col3.metric("Visibility", f"{visibility} mi")
col4.metric("Recommended Signal Action", signal_action)

st.markdown("---")

# Gauge & Visuals
c1, c2 = st.columns(2)
with c1:
    st.subheader("Machine Learning Prediction Severity Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score if predicted_class != 0 else (100 - risk_score),
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color_code},
            'steps': [
                {'range': [0, 38], 'color': '#d4edda'},
                {'range': [38, 60], 'color': '#fff3cd'},
                {'range': [60, 100], 'color': '#f8d7da'}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Hourly Traffic Density Trend")
    hourly_df = df.groupby("Hour")["Vehicle_Density"].mean().reset_index()
    fig_line = px.area(hourly_df, x="Hour", y="Vehicle_Density")
    st.plotly_chart(fig_line, use_container_width=True)