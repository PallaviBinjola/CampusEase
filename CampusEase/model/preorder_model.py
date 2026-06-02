import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the trained LSTM model and scaler
model = load_model("model/food_demand_lstm.h5")
scaler = joblib.load("model/scaler.pkl")

def predict_preorder(center_id, meal_id, checkout_price, base_price, emailer_for_promotion, homepage_featured):
    # Prepare input as array
    data = np.array([[center_id, meal_id, checkout_price, base_price,
                      emailer_for_promotion, homepage_featured]])

    # Scale the data using saved scaler
    scaled_data = scaler.transform(data)

    # Reshape for LSTM [samples, time_steps, features]
    scaled_data = np.reshape(scaled_data, (scaled_data.shape[0], 1, scaled_data.shape[1]))

    # Predict demand
    predicted_demand = model.predict(scaled_data)

    # Return predicted demand as float
    return float(predicted_demand[0][0])
