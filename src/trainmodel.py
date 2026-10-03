import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 1. Load Dataset
data_path = os.path.join('data', 'traffic_data.csv')
df = pd.read_csv(data_path)

# 2. Define Features (X) and Target (y)
X = df[['Hour', 'DayOfWeek', 'Temperature_F', 'Humidity_Pct', 
        'Visibility_mi', 'Wind_Speed_mph', 'Vehicle_Density', 
        'Traffic_Signal', 'Crossing']]
y = df['Accident_Risk_Level']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Train XGBoost Model
print("Training XGBoost Classifier...")
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Save Model Artifacts
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgb_traffic_model.pkl')
print("Model saved to 'models/xgb_traffic_model.pkl'!")