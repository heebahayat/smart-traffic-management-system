import pandas as pd
import numpy as np
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

np.random.seed(42)
n_samples = 1000

# Generate realistic synthetic traffic features
hours = np.random.randint(0, 24, size=n_samples)
day_of_week = np.random.randint(0, 7, size=n_samples)
temperature = np.random.uniform(50, 100, size=n_samples)  # in Fahrenheit
humidity = np.random.uniform(20, 100, size=n_samples)
visibility = np.random.uniform(0.5, 10.0, size=n_samples) # in miles
wind_speed = np.random.uniform(0, 25, size=n_samples)
vehicle_density = np.random.randint(5, 120, size=n_samples) # vehicles per lane

traffic_signal = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
crossing = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

# Rule-based calculation for accident risk score (0: Low, 1: Medium, 2: High)
risk_score = (
    (vehicle_density * 0.03) +
    (10 - visibility) * 0.4 +
    (wind_speed * 0.1) +
    np.where((hours >= 7) & (hours <= 10) | (hours >= 17) & (hours <= 20), 2, 0)
)

# Bin into 3 risk levels
accident_risk = pd.qcut(risk_score, q=3, labels=[0, 1, 2]).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Hour': hours,
    'DayOfWeek': day_of_week,
    'Temperature_F': np.round(temperature, 1),
    'Humidity_Pct': np.round(humidity, 1),
    'Visibility_mi': np.round(visibility, 1),
    'Wind_Speed_mph': np.round(wind_speed, 1),
    'Vehicle_Density': vehicle_density,
    'Traffic_Signal': traffic_signal,
    'Crossing': crossing,
    'Accident_Risk_Level': accident_risk
})

# Save to data folder
output_path = 'data/traffic_data.csv'
df.to_csv(output_path, index=False)
print(f"Dataset successfully created at '{output_path}' with {len(df)} records!")