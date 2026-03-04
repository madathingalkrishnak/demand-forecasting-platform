import pandas as pd
import numpy as np
import mlflow
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

os.makedirs('models/prophet', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)

MLFLOW_URI  = "http://localhost:5001"
EXPERIMENT  = "demand-forecasting-v2"
TARGET      = "order_count"
TEST_DAYS   = 14

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet('data/processed/features.parquet')
df = df.sort_values('hour_ts').reset_index(drop=True)
df['is_raining'] = df['is_raining'].astype(float)

def evaluate(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    print(f"  {label:25s} MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.1f}%")
    return {'mae': mae, 'rmse': rmse, 'mape': mape}

def train_test_split_temporal(data, test_days=TEST_DAYS):
    cutoff = data['hour_ts'].max() - pd.Timedelta(days=test_days)
    return data[data['hour_ts'] <= cutoff], data[data['hour_ts'] > cutoff]

# ── Train Prophet per zone, collect results ───────────────────────────────────
print("\n" + "="*55)
print("PROPHET — Per-Zone Forecasting")
print("="*55)

zones         = df['zone_id'].unique()
all_maes      = []
zone_results  = {}
DEMO_ZONE     = "zone_003"

for zone_id in sorted(zones):
    zone_df = df[df['zone_id'] == zone_id][['hour_ts', TARGET, 'temperature',
                                             'is_raining', 'is_weekend']].copy()

    # Prophet requires columns named ds and y
    #zone_df = zone_df.rename(columns={'hour_ts': 'ds', TARGET: 'y'})
    for col in ['temperature', 'is_raining', 'is_weekend']:
        zone_df[col] = zone_df[col].ffill().fillna(0)
    train, test = train_test_split_temporal(zone_df)


    # Prophet requires columns named ds and y
    train = train.rename(columns={'hour_ts': 'ds', TARGET: 'y'})
    test  = test.rename(columns={'hour_ts': 'ds', TARGET: 'y'})

    # ── Build model ───────────────────────────────────────────────────────────
    m = Prophet(
        changepoint_prior_scale=0.05,   # flexibility of trend
        seasonality_prior_scale=10,     # strength of seasonality
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,       # only 6 months of data
        interval_width=0.80
    )

    # Add weather regressors
    m.add_regressor('temperature')
    m.add_regressor('is_raining')
    m.add_regressor('is_weekend')

    m.fit(train)

    # Predict on test set
    future   = test[['ds', 'temperature', 'is_raining', 'is_weekend']].copy()
    forecast = m.predict(future)

    y_true = test['y'].values
    y_pred = np.maximum(0, forecast['yhat'].values)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    all_maes.append(mae)
    zone_results[zone_id] = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'test': test,
        'forecast': forecast,
        'model': m
    }

    status = "🎓" if 'university' in str(df[df['zone_id']==zone_id]['zone_type_university'].max()) else ""
    print(f"  {zone_id} {status}: MAE={mae:.2f}")

avg_mae = np.mean(all_maes)
avg_rmse = np.mean([v['rmse'] for v in zone_results.values()])
avg_mape = np.mean([v['mape'] for v in zone_results.values()])
print(f"\n  Average MAE across all zones: {avg_mae:.2f}")
print(f"  Average RMSE: {avg_rmse:.2f}")
print(f"  Average MAPE: {avg_mape:.1f}%")

# ── Log to MLflow ─────────────────────────────────────────────────────────────
with mlflow.start_run(run_name="prophet_per_zone"):
    mlflow.log_param('model_type', 'Prophet')
    mlflow.log_param('changepoint_prior_scale', 0.05)
    mlflow.log_param('seasonality_prior_scale', 10)
    mlflow.log_param('regressors', 'temperature,is_raining,is_weekend')
    mlflow.log_param('test_days', TEST_DAYS)
    mlflow.log_param('zones', len(zones))
    mlflow.log_metric('mae', avg_mae)
    mlflow.log_metric('mae_hardest_zone', zone_results[DEMO_ZONE]['mae'])

# ── Plot: Prophet components for demo zone ────────────────────────────────────
print(f"\nGenerating Prophet component plots for {DEMO_ZONE}...")
demo = zone_results[DEMO_ZONE]

fig = demo['model'].plot_components(demo['forecast'])
plt.suptitle(f'Prophet Seasonality Components — {DEMO_ZONE}', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'notebooks/06_prophet_components_{DEMO_ZONE}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Saved: notebooks/06_prophet_components_{DEMO_ZONE}.png")

# ── Plot: Forecast vs Actual for demo zone ────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
fig.suptitle(f'Prophet Forecast vs Actual — {DEMO_ZONE} (Last 14 Days)', fontweight='bold')

test_df   = demo['test']
fore_df   = demo['forecast']
sample_n  = 7 * 24  # last 7 days

ax = axes[0]
ax.plot(test_df['ds'].values[-sample_n:], test_df['y'].values[-sample_n:],
        label='Actual', color='#2c3e50', linewidth=1.5)
ax.plot(fore_df['ds'].values[-sample_n:], fore_df['yhat'].values[-sample_n:],
        label='Prophet', color='#9b59b6', linewidth=1.5, linestyle='--')
ax.fill_between(fore_df['ds'].values[-sample_n:],
                fore_df['yhat_lower'].values[-sample_n:],
                fore_df['yhat_upper'].values[-sample_n:],
                alpha=0.2, color='#9b59b6', label='80% CI')
ax.set_title('Hourly Predictions (last 7 days)', fontweight='bold')
ax.set_ylabel('Orders/Hour')
ax.legend()
ax.grid(alpha=0.3)

# Daily aggregated comparison
ax = axes[1]
test_df   = test_df.copy()
fore_df   = fore_df.copy()
test_df['date']  = pd.to_datetime(test_df['ds']).dt.date
fore_df['date']  = pd.to_datetime(fore_df['ds']).dt.date

daily_actual = test_df.groupby('date')['y'].sum()
daily_pred   = fore_df.groupby('date')['yhat'].sum().clip(lower=0)

x = range(len(daily_actual))
ax.bar([i - 0.2 for i in x], daily_actual.values, width=0.4,
       label='Actual', color='#2c3e50')
ax.bar([i + 0.2 for i in x], daily_pred.values, width=0.4,
       label='Prophet', color='#9b59b6', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(daily_actual.index, rotation=45, ha='right')
ax.set_title('Daily Aggregated Demand', fontweight='bold')
ax.set_ylabel('Total Orders/Day')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/07_prophet_forecast_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: notebooks/07_prophet_forecast_vs_actual.png")

# ── Final comparison table ────────────────────────────────────────────────────
print("\n" + "="*55)
print("FULL MODEL COMPARISON")
print("="*55)
print(f"\n  {'Model':<30} {'MAE':>6}  {'Notes'}")
print(f"  {'-'*55}")
print(f"  {'Naive (same hour last week)':<30} {'2.91':>6}  baseline")
print(f"  {'Prophet (per zone avg)':<30} {avg_mae:>6.2f}  captures seasonality, CI")
print(f"  {'XGBoost (single zone)':<30} {'2.31':>6}  tabular features")
print(f"  {'XGBoost (global all zones)':<30} {'1.61':>6}  best overall ✅")
print(f"\n  Prophet MAE on {DEMO_ZONE}: {zone_results[DEMO_ZONE]['mae']:.2f}")
print(f"  XGBoost MAE on {DEMO_ZONE}: 2.21  (from previous run)")
print(f"\n  → XGBoost wins on accuracy")
print(f"  → Prophet wins on interpretability + confidence intervals")
print(f"  → In production: use XGBoost for predictions, Prophet for anomaly detection")