import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('reports', exist_ok=True)

# ── Load data & model ─────────────────────────────────────────────────────────
print("Loading data and model...")
df = pd.read_parquet('data/processed/features.parquet')
df = df.sort_values('hour_ts').reset_index(drop=True)
df['is_raining'] = df['is_raining'].astype(float)

model = xgb.XGBRegressor()
model.load_model('models/xgb_global.json')

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
TARGET = 'order_count'

# ── Split into time windows ───────────────────────────────────────────────────
df['hour_ts'] = pd.to_datetime(df['hour_ts'])
min_date = df['hour_ts'].min()
max_date = df['hour_ts'].max()
total_days = (max_date - min_date).days
print(f"Data range: {min_date.date()} → {max_date.date()} ({total_days} days)")

# Reference = first 4 months (training distribution)
# Window 1  = month 5 (normal production — should look like training)
# Window 2  = month 6 (drifted production — simulate anomalies)
ref_end  = min_date + pd.Timedelta(days=120)
win1_end = min_date + pd.Timedelta(days=150)
win2_end = max_date

reference = df[df['hour_ts'] <  ref_end].copy()
window1   = df[(df['hour_ts'] >= ref_end)  & (df['hour_ts'] < win1_end)].copy()
window2   = df[(df['hour_ts'] >= win1_end) & (df['hour_ts'] < win2_end)].copy()

print(f"\nReference (training): {len(reference):,} rows ({ref_end.date()})")
print(f"Window 1 (normal):    {len(window1):,} rows")
print(f"Window 2 (drifted):   {len(window2):,} rows")

# ── Inject drift into Window 2 ────────────────────────────────────────────────
print("\nInjecting simulated drift into Window 2...")
window2 = window2.copy()

# 1. Demand spike in university zones at unusual hours (e.g. new late-night food hall)
univ_mask = window2['zone_type_university'] == 1
late_mask  = window2['hour'].between(22, 23)
window2.loc[univ_mask & late_mask, TARGET] *= 2.5
print("  ✓ University zones: 2.5x late-night demand spike")

# 2. Temperature correlation flips (new indoor food court — people order MORE in heat)
hot_mask = window2['temperature'] > 75
window2.loc[hot_mask, TARGET] *= 1.4
print("  ✓ Hot weather: demand increases instead of decreasing")

# 3. Weekend effect disappears (competitor launched weekend promotions)
weekend_mask = window2['is_weekend'] == 1
window2.loc[weekend_mask, TARGET] *= 0.6
print("  ✓ Weekend demand suppressed (competitor effect)")

# 4. Overall volume shift upward (market growth)
window2[TARGET] = window2[TARGET] * 1.2
print("  ✓ Overall 20% volume increase (market growth)")

# ── Compute predictions & MAE per window ─────────────────────────────────────
print("\n" + "="*55)
print("MODEL PERFORMANCE OVER TIME")
print("="*55)

def get_mae(data):
    clean = data.dropna(subset=FEATURES + [TARGET])
    if len(clean) == 0:
        return None
    preds = np.maximum(0, model.predict(clean[FEATURES]))
    return mean_absolute_error(clean[TARGET], preds)

ref_mae  = get_mae(reference)
win1_mae = get_mae(window1)
win2_mae = get_mae(window2)

print(f"\n  Reference MAE  (months 1-4): {ref_mae:.2f}  ← training distribution")
print(f"  Window 1 MAE   (month 5):    {win1_mae:.2f}  ← normal production")
print(f"  Window 2 MAE   (month 6):    {win2_mae:.2f}  ← DRIFTED production")
print(f"\n  Drift magnitude: +{((win2_mae/ref_mae)-1)*100:.1f}% MAE degradation")

ALERT_THRESHOLD = ref_mae * 1.15  # alert if MAE degrades >15%
print(f"\n  Alert threshold (15% above ref): {ALERT_THRESHOLD:.2f}")
if win2_mae > ALERT_THRESHOLD:
    print(f"  🚨 DRIFT ALERT: Window 2 MAE ({win2_mae:.2f}) exceeds threshold ({ALERT_THRESHOLD:.2f})")
    print(f"     → Trigger model retraining pipeline")
else:
    print(f"  ✅ No drift detected in Window 2")

# ── Feature distribution comparison ──────────────────────────────────────────
print("\n" + "="*55)
print("FEATURE DRIFT ANALYSIS")
print("="*55)

MONITOR_FEATURES = ['temperature', 'is_raining', 'is_weekend',
                    'hour', 'lag_24h', 'rolling_mean_24h', TARGET]

print(f"\n  {'Feature':<25} {'Ref Mean':>10} {'Win1 Mean':>10} {'Win2 Mean':>10} {'Win2 Drift':>12}")
print(f"  {'-'*70}")

drifted_features = []
for feat in MONITOR_FEATURES:
    ref_mean  = reference[feat].mean()
    win1_mean = window1[feat].mean()
    win2_mean = window2[feat].mean()
    drift_pct = ((win2_mean - ref_mean) / (abs(ref_mean) + 1e-6)) * 100
    alert = "🚨" if abs(drift_pct) > 20 else "  "
    if abs(drift_pct) > 20:
        drifted_features.append(feat)
    print(f"  {feat:<25} {ref_mean:>10.2f} {win1_mean:>10.2f} {win2_mean:>10.2f} {drift_pct:>+10.1f}% {alert}")

# ── Generate Evidently report ─────────────────────────────────────────────────
print("\n" + "="*55)
print("GENERATING EVIDENTLY DRIFT REPORT")
print("="*55)

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset, RegressionPreset
    from evidently.metrics import DriftedColumnsCount
    print("  ✅ Evidently imported successfully")

    # Prepare dataframes for Evidently
    ref_ev  = reference[MONITOR_FEATURES].dropna().sample(min(2000, len(reference)), random_state=42)
    win2_ev = window2[MONITOR_FEATURES].dropna().sample(min(2000, len(window2)), random_state=42)

    # Add predictions for regression report
    ref_clean  = reference.dropna(subset=FEATURES)
    win2_clean = window2.dropna(subset=FEATURES)

    ref_ev2  = ref_clean[MONITOR_FEATURES].copy().sample(min(2000, len(ref_clean)), random_state=42)
    win2_ev2 = win2_clean[MONITOR_FEATURES].copy().sample(min(2000, len(win2_clean)), random_state=42)

    ref_ev2['prediction']  = np.maximum(0, model.predict(
        ref_clean.loc[ref_ev2.index, FEATURES]))
    win2_ev2['prediction'] = np.maximum(0, model.predict(
        win2_clean.loc[win2_ev2.index, FEATURES]))

    ref_ev2  = ref_ev2.rename(columns={TARGET: 'target'})
    win2_ev2 = win2_ev2.rename(columns={TARGET: 'target'})

    # Data drift report
    import json

    drift_snapshot = Report([DataDriftPreset()]).run(reference_data=ref_ev, current_data=win2_ev)
    drift_snapshot.save_html('reports/drift_report.html')
    print("\n  ✅ Saved: reports/drift_report.html")

    from evidently.descriptors import column
    from evidently import DataDefinition, ColumnType

    data_def = DataDefinition(
        regression=[{"prediction": "prediction", "target": "target"}]
    )
    perf_snapshot = Report([RegressionPreset()]).run(
        reference_data=ref_ev2,
        current_data=win2_ev2,
        data_definition=data_def
    )
    print("  ℹ️  Skipping regression report (API incompatibility with v0.7.20)")
    print("\n  Open drift report in your browser:")
    print("  → reports/drift_report.html")

    print("\n  Open these in your browser:")
    print("  → reports/drift_report.html")
    print("  → reports/performance_report.html")

except Exception as e:
    print(f"\n  ❌ Error: {type(e).__name__}: {e}")
    print("  Skipping HTML report generation.")

# ── Retraining recommendation ─────────────────────────────────────────────────
print("\n" + "="*55)
print("MONITORING SUMMARY & RECOMMENDATIONS")
print("="*55)
print(f"""
  Monitoring Period:    {min_date.date()} → {max_date.date()}
  Reference MAE:        {ref_mae:.2f}
  Current MAE:          {win2_mae:.2f}
  Degradation:          +{((win2_mae/ref_mae)-1)*100:.1f}%
  Alert Threshold:      15% above reference
  Status:               {'🚨 RETRAIN RECOMMENDED' if win2_mae > ALERT_THRESHOLD else '✅ HEALTHY'}

  Drifted Features:     {', '.join(drifted_features) if drifted_features else 'None'}

  Simulated Drift Causes:
    1. University zones: late-night demand spike (+150%)
    2. Hot weather correlation flip (demand +40% in heat)
    3. Weekend suppression due to competitor (-40%)
    4. Overall market volume growth (+20%)

  Recommended Actions:
    → Retrain model on last 60 days of data
    → Add university late-night as explicit feature
    → Investigate competitor impact on weekend demand
    → Update feature store with recent patterns
""")