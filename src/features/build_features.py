import pandas as pd
import numpy as np

def build_features(orders_path='/Users/krishnakishore/Documents/demand-forecasting-platform/data/raw/orders.csv',
                   weather_path='/Users/krishnakishore/Documents/demand-forecasting-platform/data/raw/weather.csv',
                   zones_path='/Users/krishnakishore/Documents/demand-forecasting-platform/data/raw/zones.csv',
                   output_path='/Users/krishnakishore/Documents/demand-forecasting-platform/data/processed/features.parquet'):

    print("Loading data...")
    orders  = pd.read_csv(orders_path, parse_dates=['timestamp'])
    weather = pd.read_csv(weather_path, parse_dates=['timestamp'])
    zones   = pd.read_csv(zones_path)

    # ── Step 1: Aggregate to zone-hour level ──────────────────────────────────
    orders['hour_ts'] = orders['timestamp'].dt.floor('h')
    zone_hour = (orders.groupby(['zone_id', 'hour_ts'])
                       .agg(order_count=('order_id', 'count'),
                            avg_order_value=('order_value', 'mean'))
                       .reset_index())

    # Create complete grid (every zone × every hour, fill 0s for missing)
    all_zones = orders['zone_id'].unique()
    all_hours = pd.date_range(orders['hour_ts'].min(),
                              orders['hour_ts'].max(), freq='h')
    grid = pd.MultiIndex.from_product([all_zones, all_hours],
                                       names=['zone_id','hour_ts'])
    zone_hour = (zone_hour.set_index(['zone_id','hour_ts'])
                          .reindex(grid, fill_value=0)
                          .reset_index())

    # ── Step 2: Time-based features ───────────────────────────────────────────
    zone_hour['hour']          = zone_hour['hour_ts'].dt.hour
    zone_hour['dayofweek']     = zone_hour['hour_ts'].dt.dayofweek
    zone_hour['is_weekend']    = zone_hour['dayofweek'].isin([5, 6]).astype(int)
    zone_hour['month']         = zone_hour['hour_ts'].dt.month
    zone_hour['is_lunch_rush'] = zone_hour['hour'].between(11, 14).astype(int)
    zone_hour['is_dinner_rush']= zone_hour['hour'].between(17, 21).astype(int)
    zone_hour['is_late_night'] = zone_hour['hour'].isin([22, 23, 0, 1]).astype(int)

    # Cyclical encoding (so hour 23 and hour 0 are close together)
    zone_hour['hour_sin']  = np.sin(2 * np.pi * zone_hour['hour'] / 24)
    zone_hour['hour_cos']  = np.cos(2 * np.pi * zone_hour['hour'] / 24)
    zone_hour['dow_sin']   = np.sin(2 * np.pi * zone_hour['dayofweek'] / 7)
    zone_hour['dow_cos']   = np.cos(2 * np.pi * zone_hour['dayofweek'] / 7)

    # ── Step 3: Lag features (the most important for forecasting) ─────────────
    print("Building lag features...")
    zone_hour = zone_hour.sort_values(['zone_id', 'hour_ts'])

    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:  # 168 = 1 week ago
        zone_hour[f'lag_{lag}h'] = (zone_hour.groupby('zone_id')['order_count']
                                             .shift(lag))

    # Rolling averages
    for window in [3, 6, 24]:
        zone_hour[f'rolling_mean_{window}h'] = (
            zone_hour.groupby('zone_id')['order_count']
                     .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean()))

    # Same hour last week (very powerful for food delivery)
    zone_hour['same_hour_last_week'] = (zone_hour.groupby(['zone_id', 'hour'])['order_count']
                                                  .shift(7))

    # ── Step 4: Weather features ──────────────────────────────────────────────
    print("Merging weather...")
    weather['hour_ts'] = weather['timestamp'].dt.floor('h')
    weather_agg = (weather.groupby('hour_ts')
                          .agg(temperature=('temperature', 'mean'),
                               precipitation=('precipitation', 'sum'),
                               is_raining=('is_raining', 'max'),
                               wind_speed=('wind_speed', 'mean'))
                          .reset_index())

    zone_hour = zone_hour.merge(weather_agg, on='hour_ts', how='left')

    # ── Step 5: Zone static features ─────────────────────────────────────────
    zones_encoded = pd.get_dummies(zones[['zone_id','zone_type']], columns=['zone_type'])
    zone_hour = zone_hour.merge(zones_encoded, on='zone_id', how='left')

    # Zone demand rank (is this a high or low demand zone?)
    zone_avg = (zone_hour.groupby('zone_id')['order_count']
                         .mean()
                         .rank(pct=True)
                         .rename('zone_demand_percentile'))
    zone_hour = zone_hour.merge(zone_avg, on='zone_id', how='left')

    # ── Step 6: Drop rows with NaN lags (first week of data) ─────────────────
    zone_hour = zone_hour.dropna(subset=['lag_168h'])

    # ── Save ──────────────────────────────────────────────────────────────────
    import os
    os.makedirs('data/processed', exist_ok=True)
    zone_hour.to_parquet(output_path, index=False)

    print(f"\n✅ Features saved to {output_path}")
    print(f"   Shape: {zone_hour.shape}")
    print(f"   Date range: {zone_hour['hour_ts'].min()} → {zone_hour['hour_ts'].max()}")
    print(f"   Zones: {zone_hour['zone_id'].nunique()}")
    print(f"\n   Feature columns ({len(zone_hour.columns)}):")
    for col in sorted(zone_hour.columns):
        print(f"     {col}")

    return zone_hour

if __name__ == "__main__":
    df = build_features()