import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)
n = 5000
vehicle_speed = np.random.normal(60, 15, n).clip(0, 150)
driver_age = np.random.normal(35, 12, n).clip(16, 90)
alcohol_level = np.random.exponential(0.2, n)
traffic_density = np.random.choice(['low','medium','high'], n, p=[0.4,0.4,0.2])
weather = np.random.choice(['clear','rain','fog','snow'], n, p=[0.7,0.18,0.09,0.03])
road_light = np.random.choice(['daylight','dark_no_lights','dark_lights_on'], n, p=[0.75,0.05,0.2])
vehicle_type = np.random.choice(['car','motorcycle','truck','bus'], n, p=[0.7,0.15,0.1,0.05])
seatbelt = np.random.choice([0,1], n, p=[0.15,0.85])
road_surface = np.random.choice(['dry','wet','icy'], n, p=[0.82,0.16,0.02])
road_alignment = np.random.choice(['straight','curve'], n, p=[0.8,0.2])
hour = np.random.randint(0, 24, n)

# Target variable (severity)
severity = (
    0.7 * vehicle_speed
    + 5.0 * (alcohol_level > 0.3) * alcohol_level * 20
    - 8.0 * seatbelt
    + 7.0 * (vehicle_type == 'motorcycle')
    + 5.0 * (vehicle_type == 'truck')
    + 3.0 * (weather == 'rain')
    + 8.0 * (weather == 'fog')
    + 12.0 * (weather == 'snow')
    + 6.0 * (road_surface == 'wet')
    + 18.0 * (road_surface == 'icy')
    + 10.0 * (road_light == 'dark_no_lights')
    + 4.0 * (road_alignment == 'curve')
    + np.where(traffic_density == 'high', -4.0, np.where(traffic_density == 'low', 2.0, 0.0))
    + 2.0 * ((hour < 6) | (hour > 22))
)
severity = np.clip(severity + np.random.normal(0, 8, n), 0, 100)

# Save dataset
df = pd.DataFrame({
    'vehicle_speed': vehicle_speed, 'driver_age': driver_age, 'alcohol_level': alcohol_level,
    'traffic_density': traffic_density, 'weather': weather, 'road_light': road_light,
    'vehicle_type': vehicle_type, 'seatbelt': seatbelt, 'road_surface': road_surface,
    'road_alignment': road_alignment, 'hour': hour, 'severity': severity
})
df.to_csv('synthetic_accidents.csv', index=False)

# Train model
target = 'severity'
numeric = ['vehicle_speed', 'driver_age', 'alcohol_level', 'hour']
categorical = ['traffic_density','weather','road_light','vehicle_type','seatbelt','road_surface','road_alignment']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
joblib.dump(model, 'accident_severity_model.joblib')

# Evaluate and plot
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel('Actual Severity')
plt.ylabel('Predicted Severity')
plt.title('Actual vs Predicted Severity')
plt.savefig('actual_vs_predicted.png')
print("Model, dataset, and plot saved.")
