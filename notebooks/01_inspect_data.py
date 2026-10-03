import pandas as pd
import os

# Resolves correct path whether run from root or inside notebooks/
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'traffic_data.csv')

df = pd.read_csv(data_path)

print("--- Dataset Shape ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

print("--- First 5 Rows ---")
print(df.head())

print("\n--- Accident Risk Class Distribution ---")
print(df['Accident_Risk_Level'].value_counts())