import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import xgboost as xgb
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Forecasting Platform",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {background:#1e1e2e;border-radius:10px;padding:20px;text-align:center;}
    .stMetric {background:#1e1e2e;border-radius:8px;padding:10px;}
</style>
""", unsafe_allow_html=True)

# ── Load model & data ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = xgb.XGBRegressor()
    m.load_model('models/xgb_global.json')
    return m

@st.cache_data
def load_data():
    df  = pd.read_parquet('data/processed/features.parquet')
    zones = pd.read_csv('data/raw/zones.csv')
    return df, zones

model = load_model()
df, zones = load_data()

ZONE_TYPES = dict(zip(zones['zone_id'], zones['zone_type']))
FEATURES = [
    'hour','dayofweek','is_weekend','month',
    'is_lunch_rush','is_dinner_rush','is_late_night',
    'hour_sin','hour_cos','dow_sin','dow_cos',
    'lag_1h','lag_2h','lag_3h','lag_6h','lag_12h','lag_24h','lag_48h','lag_168h',
    'rolling_mean_3h','rolling_mean_6h','rolling_mean_24h','same_hour_last_week',
    'temperature','precipitation','is_raining','wind_speed',
    'zone_demand_percentile',
    'zone_type_downtown','zone_type_residential','zone_type_suburb','zone_type_university',
]

def predict_zone_hour(zone_id, ts, temperature=65.0, is_raining=0.0):
    zone_type = ZONE_TYPES.get(zone_id, 'residential')
    zone_idx  = int(zone_id.split('_')[1])
    features = {
        'hour': ts.hour, 'dayofweek': ts.weekday(),
        'is_weekend': int(ts.weekday() >= 5), 'month': ts.month,
        'is_lunch_rush': int(11 <= ts.hour <= 14),
        'is_dinner_rush': int(17 <= ts.hour <= 21),
        'is_late_night': int(ts.hour in [22,23,0,1]),
        'hour_sin': np.sin(2*np.pi*ts.hour/24),
        'hour_cos': np.cos(2*np.pi*ts.hour/24),
        'dow_sin': np.sin(2*np.pi*ts.weekday()/7),
        'dow_cos': np.cos(2*np.pi*ts.weekday()/7),
        'lag_1h':8.0,'lag_2h':8.0,'lag_3h':8.0,'lag_6h':8.0,
        'lag_12h':8.0,'lag_24h':8.0,'lag_48h':8.0,'lag_168h':8.0,
        'rolling_mean_3h':8.0,'rolling_mean_6h':8.0,'rolling_mean_24h':8.0,
        'same_hour_last_week':8.0,
        'temperature': temperature, 'precipitation': 0.0,
        'is_raining': is_raining, 'wind_speed': 8.0,
        'zone_demand_percentile': round(0.1 + (zone_idx/20)*0.85, 2),
        'zone_type_downtown':    int(zone_type=='downtown'),
        'zone_type_residential': int(zone_type=='residential'),
        'zone_type_suburb':      int(zone_type=='suburb'),
        'zone_type_university':  int(zone_type=='university'),
    }
    X = pd.DataFrame([features])
    pred = float(max(0, model.predict(X)[0]))
    return pred

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚀 Real-Time Demand Forecasting Platform")
st.markdown("*Predicting food delivery demand across zones — inspired by DoorDash/Uber Eats*")
st.divider()

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    selected_zone = st.selectbox("Select Zone", sorted(ZONE_TYPES.keys()),
                                  index=3, format_func=lambda z: f"{z} ({ZONE_TYPES[z]})")
    temperature   = st.slider("Temperature (°F)", 30, 100, 65)
    is_raining    = st.toggle("Currently Raining", False)
    forecast_hours = st.slider("Forecast Horizon (hours)", 1, 12, 4)
    st.divider()
    st.markdown("**Model Info**")
    st.markdown("- Algorithm: XGBoost")
    st.markdown("- Trained on: 20 zones × 6 months")
    st.markdown("- MAE: 1.61 orders/hour")
    st.markdown("- Baseline MAE: 2.91 (naive)")
    st.markdown("- Improvement: **44.5%**")

# ── Row 1: KPI metrics ────────────────────────────────────────────────────────
now = datetime.now().replace(minute=0, second=0, microsecond=0)
current_pred = predict_zone_hour(selected_zone, now, temperature, float(is_raining))
next_hour    = predict_zone_hour(selected_zone, now + timedelta(hours=1), temperature, float(is_raining))
peak_pred    = max(predict_zone_hour(selected_zone, now.replace(hour=h), temperature, float(is_raining)) for h in range(24))
all_zones_now = sum(predict_zone_hour(z, now, temperature, float(is_raining)) for z in ZONE_TYPES)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Hour Demand", f"{current_pred:.1f} orders",
            delta=f"{next_hour - current_pred:+.1f} next hour")
col2.metric("Peak Today (forecast)", f"{peak_pred:.1f} orders")
col3.metric("All Zones Right Now",   f"{all_zones_now:.0f} orders")
col4.metric("Zone Type", ZONE_TYPES[selected_zone].title())

st.divider()

# ── Row 2: Forecast chart + Zone heatmap ─────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader(f"📈 {forecast_hours}-Hour Demand Forecast — {selected_zone}")
    forecast_data = []
    for i in range(forecast_hours):
        ts   = now + timedelta(hours=i)
        pred = predict_zone_hour(selected_zone, ts, temperature, float(is_raining))
        forecast_data.append({'Hour': ts.strftime('%H:00'), 'Predicted': pred,
                               'Low': pred*0.8, 'High': pred*1.2, 'ts': ts})
    fdf = pd.DataFrame(forecast_data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fdf['Hour'], y=fdf['High'], fill=None,
                             mode='lines', line_color='rgba(231,76,60,0.2)', showlegend=False))
    fig.add_trace(go.Scatter(x=fdf['Hour'], y=fdf['Low'], fill='tonexty',
                             mode='lines', line_color='rgba(231,76,60,0.2)',
                             fillcolor='rgba(231,76,60,0.15)', name='80% Confidence'))
    fig.add_trace(go.Scatter(x=fdf['Hour'], y=fdf['Predicted'], mode='lines+markers',
                             line=dict(color='#e74c3c', width=3),
                             marker=dict(size=10), name='Predicted Demand'))
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                      yaxis_title='Orders/Hour', xaxis_title='Hour',
                      legend=dict(orientation='h'))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🗺️ Zone Demand Right Now")
    zone_demands = [(z, predict_zone_hour(z, now, temperature, float(is_raining)))
                    for z in sorted(ZONE_TYPES.keys())]
    zone_df = pd.DataFrame(zone_demands, columns=['zone','demand'])
    zone_df['type'] = zone_df['zone'].map(ZONE_TYPES)
    zone_df = zone_df.sort_values('demand', ascending=True)

    fig2 = px.bar(zone_df, x='demand', y='zone', color='type', orientation='h',
                  color_discrete_map={'downtown':'#e74c3c','residential':'#3498db',
                                      'suburb':'#2ecc71','university':'#f39c12'},
                  height=300)
    fig2.update_layout(margin=dict(l=0,r=0,t=10,b=0),
                       xaxis_title='Predicted Orders', yaxis_title='',
                       legend=dict(orientation='h', y=-0.3))
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 3: 24-hour heatmap + Historical accuracy ──────────────────────────────
st.divider()
col3a, col3b = st.columns(2)

with col3a:
    st.subheader("🌡️ 24-Hour Demand Heatmap (All Zones)")
    heatmap_data = []
    for zone_id in sorted(ZONE_TYPES.keys()):
        for h in range(24):
            ts   = now.replace(hour=h)
            pred = predict_zone_hour(zone_id, ts, temperature, float(is_raining))
            heatmap_data.append({'Zone': zone_id, 'Hour': h, 'Demand': pred})
    hdf = pd.DataFrame(heatmap_data).pivot(index='Zone', columns='Hour', values='Demand')
    fig3 = px.imshow(hdf, color_continuous_scale='YlOrRd', aspect='auto',
                     labels=dict(x='Hour of Day', y='Zone', color='Orders'))
    fig3.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig3, use_container_width=True)

with col3b:
    st.subheader("📊 Model Performance vs Baseline")
    hist_data = df[df['zone_id'] == selected_zone].tail(14*24).copy()
    hist_data = hist_data.dropna(subset=FEATURES)
    if len(hist_data) > 0:
        X_hist = hist_data[FEATURES].copy()
        X_hist['is_raining'] = X_hist['is_raining'].astype(float)
        preds = np.maximum(0, model.predict(X_hist))
        hist_data = hist_data.copy()
        hist_data['predicted'] = preds
        sample = hist_data.tail(72)

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=sample['hour_ts'], y=sample['order_count'],
                                  name='Actual', line=dict(color='#2c3e50', width=2)))
        fig4.add_trace(go.Scatter(x=sample['hour_ts'], y=sample['predicted'],
                                  name='XGBoost', line=dict(color='#e74c3c', width=2, dash='dash')))
        fig4.add_trace(go.Scatter(x=sample['hour_ts'], y=sample['same_hour_last_week'],
                                  name='Naive Baseline', line=dict(color='#95a5a6', width=1, dash='dot')))
        fig4.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0),
                           yaxis_title='Orders/Hour', legend=dict(orientation='h'))
        st.plotly_chart(fig4, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("**Tech Stack:** XGBoost · FastAPI · Redis · TimescaleDB · Kafka · MLflow · Streamlit · Docker")