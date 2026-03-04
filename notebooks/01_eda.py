import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────────
orders   = pd.read_csv('/Users/krishnakishore/Documents/demand-forecasting-platform/data/raw/orders.csv', parse_dates=['timestamp'])
zones    = pd.read_csv('/Users/krishnakishore/Documents/demand-forecasting-platform/data/raw/zones.csv')
weather  = pd.read_csv('/Users/krishnakishore/Documents/demand-forecasting-platform/data/raw/weather.csv', parse_dates=['timestamp'])
restaurants = pd.read_csv('/Users/krishnakishore/Documents/demand-forecasting-platform/data/raw/restaurants.csv')

print("=== Data Loaded ===")
print(f"Orders:      {len(orders):,}")
print(f"Date range:  {orders['timestamp'].min().date()} → {orders['timestamp'].max().date()}")
print(f"Zones:       {len(zones)}")
print(f"Restaurants: {len(restaurants)}")

# ── Feature extraction ─────────────────────────────────────────────────────────
orders['date']       = orders['timestamp'].dt.date
orders['hour']       = orders['timestamp'].dt.hour
orders['dayofweek']  = orders['timestamp'].dt.dayofweek  # 0=Mon
orders['day_name']   = orders['timestamp'].dt.day_name()
orders['week']       = orders['timestamp'].dt.isocalendar().week.astype(int)
orders['month']      = orders['timestamp'].dt.month

# Merge zone info
orders = orders.merge(zones[['zone_id','zone_type']], on='zone_id', how='left')

# ── Hourly demand heatmap by zone type ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Demand Patterns by Hour & Zone Type', fontsize=16, fontweight='bold')

zone_types = orders['zone_type'].unique()
colors = {'downtown': '#e74c3c', 'residential': '#3498db',
          'suburb': '#2ecc71', 'university': '#f39c12'}

ax = axes[0, 0]
hourly = orders.groupby(['hour', 'zone_type']).size().unstack()
for zt in hourly.columns:
    ax.plot(hourly.index, hourly[zt], label=zt, color=colors.get(zt), linewidth=2.5, marker='o', markersize=4)
ax.set_title('Orders by Hour of Day', fontweight='bold')
ax.set_xlabel('Hour'); ax.set_ylabel('Total Orders')
ax.legend(); ax.grid(alpha=0.3)
ax.axvspan(11, 14, alpha=0.1, color='red', label='Lunch rush')
ax.axvspan(17, 21, alpha=0.1, color='orange', label='Dinner rush')

ax = axes[0, 1]
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
daily = orders.groupby('day_name').size().reindex(day_order)
bar_colors = ['#e74c3c' if d in ['Saturday','Sunday'] else '#3498db' for d in day_order]
ax.bar(day_order, daily.values, color=bar_colors)
ax.set_title('Orders by Day of Week', fontweight='bold')
ax.set_xlabel('Day'); ax.set_ylabel('Total Orders')
ax.tick_params(axis='x', rotation=45)
ax.grid(alpha=0.3, axis='y')

ax = axes[1, 0]
pivot = orders.groupby(['dayofweek', 'hour']).size().unstack()
im = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_title('Demand Heatmap (Day × Hour)', fontweight='bold')
ax.set_xlabel('Hour of Day'); ax.set_ylabel('Day of Week')
ax.set_yticks(range(7))
ax.set_yticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels(range(0, 24, 2))
plt.colorbar(im, ax=ax, label='Order Count')

ax = axes[1, 1]
weekly = orders.groupby('week').size()
ax.plot(weekly.index, weekly.values, color='#8e44ad', linewidth=2)
ax.fill_between(weekly.index, weekly.values, alpha=0.2, color='#8e44ad')
ax.set_title('Weekly Order Volume Trend', fontweight='bold')
ax.set_xlabel('Week of Year'); ax.set_ylabel('Total Orders')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/01_demand_patterns.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: notebooks/01_demand_patterns.png")

# ── Weather correlation ────────────────────────────────────────────────────────
weather['hour'] = weather['timestamp'].dt.floor('h')
orders['hour_ts'] = orders['timestamp'].dt.floor('h')
hourly_orders = orders.groupby('hour_ts').size().reset_index(name='order_count')
weather_merge = weather.merge(hourly_orders, left_on='hour', right_on='hour_ts', how='inner')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Weather Impact on Demand', fontsize=14, fontweight='bold')

ax = axes[0]
ax.scatter(weather_merge['temperature'], weather_merge['order_count'],
           alpha=0.3, s=10, color='#e74c3c')
corr = weather_merge['temperature'].corr(weather_merge['order_count'])
ax.set_title(f'Temperature vs Orders (r={corr:.3f})', fontweight='bold')
ax.set_xlabel('Temperature (°F)'); ax.set_ylabel('Orders/Hour')

ax = axes[1]
ax.scatter(weather_merge['precipitation'], weather_merge['order_count'],
           alpha=0.3, s=10, color='#3498db')
corr2 = weather_merge['precipitation'].corr(weather_merge['order_count'])
ax.set_title(f'Precipitation vs Orders (r={corr2:.3f})', fontweight='bold')
ax.set_xlabel('Precipitation'); ax.set_ylabel('Orders/Hour')

ax = axes[2]
rain_comp = weather_merge.groupby('is_raining')['order_count'].mean()
ax.bar(['Not Raining', 'Raining'], rain_comp.values,
       color=['#f39c12', '#3498db'], width=0.5)
ax.set_title('Avg Orders: Rain vs No Rain', fontweight='bold')
ax.set_ylabel('Avg Orders/Hour')
pct_diff = (rain_comp[True] - rain_comp[False]) / rain_comp[False] * 100
ax.text(0.5, max(rain_comp.values) * 0.95,
        f'Rain effect: {pct_diff:+.1f}%',
        ha='center', fontsize=11, fontweight='bold',
        color='#2c3e50')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('notebooks/02_weather_impact.png', dpi=150, bbox_inches='tight')
print("✅ Saved: notebooks/02_weather_impact.png")

# ── Zone volatility (hardest to forecast) ─────────────────────────────────────
zone_hourly = orders.groupby(['zone_id', 'date', 'hour']).size().reset_index(name='orders')
zone_stats = zone_hourly.groupby('zone_id')['orders'].agg(['mean','std']).reset_index()
zone_stats['cv'] = zone_stats['std'] / zone_stats['mean']  # coefficient of variation
zone_stats = zone_stats.merge(zones[['zone_id','zone_type']], on='zone_id')
zone_stats = zone_stats.sort_values('cv', ascending=False)

fig, ax = plt.subplots(figsize=(14, 5))
bar_colors = [colors.get(zt, 'gray') for zt in zone_stats['zone_type']]
bars = ax.bar(zone_stats['zone_id'], zone_stats['cv'], color=bar_colors)
ax.set_title('Zone Demand Volatility (CV = Std/Mean) — Higher = Harder to Forecast',
             fontweight='bold')
ax.set_xlabel('Zone'); ax.set_ylabel('Coefficient of Variation')
ax.tick_params(axis='x', rotation=90)
ax.grid(alpha=0.3, axis='y')
ax.axhline(zone_stats['cv'].mean(), color='red', linestyle='--',
           label=f"Mean CV: {zone_stats['cv'].mean():.2f}")
ax.legend()

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[zt], label=zt) for zt in colors]
ax.legend(handles=legend_elements + [plt.Line2D([0],[0], color='red',
          linestyle='--', label=f"Mean CV: {zone_stats['cv'].mean():.2f}")])

plt.tight_layout()
plt.savefig('notebooks/03_zone_volatility.png', dpi=150, bbox_inches='tight')
print("✅ Saved: notebooks/03_zone_volatility.png")

# ── Key findings summary ───────────────────────────────────────────────────────
print("\n" + "="*50)
print("KEY EDA FINDINGS")
print("="*50)

peak_hour = orders.groupby('hour').size().idxmax()
print(f"\n📈 Peak hour overall: {peak_hour}:00")

for zt in zone_types:
    subset = orders[orders['zone_type'] == zt]
    peak = subset.groupby('hour').size().idxmax()
    print(f"   {zt:12s} peak: {peak}:00")

weekend_avg = orders[orders['is_weekend']].groupby('date').size().mean()
weekday_avg = orders[~orders['is_weekend']].groupby('date').size().mean()
print(f"\n📅 Weekend vs Weekday: {weekend_avg:.0f} vs {weekday_avg:.0f} orders/day ({(weekend_avg/weekday_avg-1)*100:+.1f}%)")

print(f"\n🌧️  Rain effect on orders: {pct_diff:+.1f}%")
print(f"🌡️  Temperature correlation: {corr:.3f}")

most_volatile = zone_stats.iloc[0]
least_volatile = zone_stats.iloc[-1]
print(f"\n🎯 Hardest zone to forecast: {most_volatile['zone_id']} ({most_volatile['zone_type']}, CV={most_volatile['cv']:.2f})")
print(f"   Easiest zone to forecast: {least_volatile['zone_id']} ({least_volatile['zone_type']}, CV={least_volatile['cv']:.2f})")

print(f"\n💰 Avg order value: ${orders['order_value'].mean():.2f}")
print(f"   Order value range: ${orders['order_value'].min():.2f} – ${orders['order_value'].max():.2f}")