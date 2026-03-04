# scripts/generate_market_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class RestaurantMarketSimulator:
    """Generate realistic food delivery market data"""
    
    def __init__(self, n_zones=20, n_restaurants=200, days=180):
        self.n_zones = n_zones
        self.n_restaurants = n_restaurants
        self.days = days
        self.start_date = datetime.now() - timedelta(days=days)
        
    def generate_zones(self):
        """Create delivery zones with characteristics"""
        zones = []
        zone_types = ['downtown', 'residential', 'suburb', 'university']
        
        for i in range(self.n_zones):
            zones.append({
                'zone_id': f'zone_{i:03d}',
                'zone_type': np.random.choice(zone_types),
                'base_demand': np.random.uniform(50, 300),
                'lat': np.random.uniform(37.7, 37.8),
                'lon': np.random.uniform(-122.5, -122.4)
            })
        
        return pd.DataFrame(zones)
    
    def generate_restaurants(self, zones_df):
        """Create restaurants assigned to zones"""
        restaurants = []
        cuisines = ['italian', 'chinese', 'mexican', 'american', 'thai', 
                   'indian', 'japanese', 'mediterranean']
        
        for i in range(self.n_restaurants):
            zone = zones_df.sample(1).iloc[0]
            restaurants.append({
                'restaurant_id': f'rest_{i:04d}',
                'zone_id': zone['zone_id'],
                'cuisine_type': np.random.choice(cuisines),
                'avg_prep_time': np.random.uniform(15, 45),
                'price_range': np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1]),
                'rating': np.random.uniform(3.5, 5.0)
            })
        
        return pd.DataFrame(restaurants)
    
    def generate_orders(self, zones_df, restaurants_df):
        """Generate order transactions with realistic patterns"""
        orders = []
        
        for day in range(self.days):
            current_date = self.start_date + timedelta(days=day)
            is_weekend = current_date.weekday() >= 5
            
            # Special events (increases demand)
            is_special = np.random.random() < 0.05  # 5% of days
            
            for hour in range(24):
                timestamp = current_date + timedelta(hours=hour)
                
                # Time-of-day patterns
                hour_multiplier = self._get_hour_multiplier(hour)
                weekend_multiplier = 1.3 if is_weekend else 1.0
                special_multiplier = 1.5 if is_special else 1.0
                
                for _, zone in zones_df.iterrows():
                    # Zone-specific patterns
                    zone_mult = self._get_zone_multiplier(zone['zone_type'], hour)
                    
                    # Expected orders this hour
                    expected_orders = (zone['base_demand'] * hour_multiplier * 
                                     weekend_multiplier * special_multiplier * zone_mult)
                    
                    # Add randomness (Poisson distribution)
                    n_orders = np.random.poisson(expected_orders / 24)  # per hour
                    
                    # Generate individual orders
                    zone_restaurants = restaurants_df[
                        restaurants_df['zone_id'] == zone['zone_id']
                    ]
                    
                    for _ in range(n_orders):
                        restaurant = zone_restaurants.sample(1).iloc[0]
                        
                        # Order details
                        order_time = timestamp + timedelta(minutes=np.random.randint(0, 60))
                        order_value = np.random.lognormal(3.5, 0.5)  # ~$30-50 avg
                        
                        orders.append({
                            'order_id': f'ord_{len(orders):08d}',
                            'timestamp': order_time,
                            'zone_id': zone['zone_id'],
                            'restaurant_id': restaurant['restaurant_id'],
                            'order_value': round(order_value, 2),
                            'prep_time': restaurant['avg_prep_time'] + np.random.normal(0, 5),
                            'delivery_time': np.random.normal(25, 8),
                            'is_weekend': is_weekend,
                            'is_special_event': is_special,
                            'hour': hour
                        })
        
        return pd.DataFrame(orders)
    
    def _get_hour_multiplier(self, hour):
        """Demand patterns by hour"""
        # Breakfast (7-9), Lunch (11-14), Dinner (17-21)
        patterns = {
            7: 0.3, 8: 0.6, 9: 0.4,
            11: 0.8, 12: 1.2, 13: 1.0, 14: 0.6,
            17: 0.9, 18: 1.4, 19: 1.5, 20: 1.2, 21: 0.7
        }
        return patterns.get(hour, 0.2)
    
    def _get_zone_multiplier(self, zone_type, hour):
        """Zone-specific patterns"""
        if zone_type == 'downtown':
            return 1.5 if 11 <= hour <= 14 else 1.0
        elif zone_type == 'university':
            return 1.3 if 19 <= hour <= 23 else 0.8
        elif zone_type == 'residential':
            return 1.2 if 17 <= hour <= 21 else 0.9
        return 1.0
    
    def generate_weather(self):
        """Generate weather data"""
        weather = []
        
        for day in range(self.days):
            current_date = self.start_date + timedelta(days=day)
            
            # Seasonal temperature patterns
            season_temp = 60 + 20 * np.sin(2 * np.pi * day / 365)
            
            for hour in range(24):
                timestamp = current_date + timedelta(hours=hour)
                
                # Daily temperature variation
                hour_adjustment = 10 * np.sin(2 * np.pi * (hour - 6) / 24)
                temp = season_temp + hour_adjustment + np.random.normal(0, 3)
                
                weather.append({
                    'timestamp': timestamp,
                    'temperature': round(temp, 1),
                    'precipitation': max(0, np.random.normal(0, 0.5)),
                    'is_raining': np.random.random() < 0.15,
                    'wind_speed': np.random.exponential(8)
                })
        
        return pd.DataFrame(weather)
    
    def generate_all(self):
        """Generate complete dataset"""
        print("Generating zones...")
        zones = self.generate_zones()
        
        print("Generating restaurants...")
        restaurants = self.generate_restaurants(zones)
        
        print("Generating orders (this may take a minute)...")
        orders = self.generate_orders(zones, restaurants)
        
        print("Generating weather data...")
        weather = self.generate_weather()
        
        return {
            'zones': zones,
            'restaurants': restaurants,
            'orders': orders,
            'weather': weather
        }

if __name__ == "__main__":
    simulator = RestaurantMarketSimulator(
        n_zones=20,
        n_restaurants=200,
        days=180  # 6 months of data
    )
    
    data = simulator.generate_all()
    
    # Save datasets
    data['zones'].to_csv('/Users/krishnakishore/Documents/demand-forecasting-platform/data/raw/zones.csv', index=False)
    data['restaurants'].to_csv('/Users/krishnakishore/Documents/demand-forecasting-platform/data/raw/restaurants.csv', index=False)
    data['orders'].to_csv('/Users/krishnakishore/Documents/demand-forecasting-platform/data/raw/orders.csv', index=False)
    data['weather'].to_csv('/Users/krishnakishore/Documents/demand-forecasting-platform/data/raw/weather.csv', index=False)
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Zones: {len(data['zones'])}")
    print(f"Restaurants: {len(data['restaurants'])}")
    print(f"Orders: {len(data['orders']):,}")
    print(f"Date range: {data['orders']['timestamp'].min()} to {data['orders']['timestamp'].max()}")
    print(f"Avg orders/day: {len(data['orders']) / 180:.0f}")
    print(f"Avg order value: ${data['orders']['order_value'].mean():.2f}")