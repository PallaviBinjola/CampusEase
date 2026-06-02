# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Example dataset (you can replace this with your own)
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6],
    'feature2': [10, 20, 30, 40, 50, 60],
    'feature3': [5, 10, 15, 20, 25, 30],
    'label': [0, 0, 1, 1, 1, 1]
})

X = data[['feature1', 'feature2', 'feature3']]
y = data['label']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'ml_model.pkl')
print("✅ Model trained and saved as ml_model.pkl")
