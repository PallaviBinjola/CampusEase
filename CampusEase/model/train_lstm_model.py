import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

print(" Loading dataset...")

# Load dataset
df = pd.read_csv(r"C:\Users\palla\CampusEase\dataset\train.csv")

print(" Data loaded successfully!")
print(df.head())

# -----------------------------
# Step 1: Data Preprocessing
# -----------------------------
print(" Preprocessing data...")

# Keep only relevant columns (adjust as needed)
df = df[['center_id', 'meal_id', 'checkout_price', 'base_price', 'emailer_for_promotion', 'homepage_featured', 'num_orders']]

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Normalize features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")

# Split into input (X) and output (y)
X = scaled_data[:, :-1]
y = scaled_data[:, -1]

# Reshape input to 3D for LSTM
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# Split train and test (80%-20%)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# Step 2: Model Building
# -----------------------------
print(" Building LSTM model...")

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# -----------------------------
# Step 3: Training
# -----------------------------
print(" Training the model...")

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# Step 4: Evaluation
# -----------------------------
print(" Evaluating model...")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# -----------------------------
# Step 5: Save Model
# -----------------------------
model.save("model/food_demand_lstm.h5")
print("\n Model training complete!")
print(" Saved model: model/food_demand_lstm.h5")
print(" Saved scaler: model/scaler.pkl")

# -----------------------------
# Step 6: Visualization
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(y_test[:100], label='Actual Demand', linewidth=2)
plt.plot(y_pred[:100], label='Predicted Demand', linewidth=2)
plt.title("Actual vs Predicted Food Demand (Sample)")
plt.xlabel("Sample Index")
plt.ylabel("Normalized Demand")
plt.legend()
plt.grid(True)
plt.show()
