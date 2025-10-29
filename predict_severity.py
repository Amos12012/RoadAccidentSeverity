import joblib
import pandas as pd

# Load the trained model
model = joblib.load('accident_severity_model.joblib')

# Define a sample accident scenario (hypothetical input)
data = pd.DataFrame([{
    'vehicle_speed': 90,          # km/h
    'driver_age': 25,             # years
    'alcohol_level': 0.4,         # blood alcohol concentration
    'traffic_density': 'high',    
    'weather': 'rain',            
    'road_light': 'dark_no_lights',
    'vehicle_type': 'motorcycle',
    'seatbelt': 0,                
    'road_surface': 'wet',
    'road_alignment': 'curve',
    'hour': 23                    # 11 PM
}])

# Predict severity
predicted_severity = model.predict(data)[0]

print("Predicted Accident Severity:", round(predicted_severity, 2))
