import joblib
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = joblib.load('models/best_random_forest_model.pkl')

# Load preprocessing artifacts
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

print("✅ Model and preprocessors loaded successfully!")
print(f"Model type: {type(model).__name__}")
print(f"Number of features: {model.n_features_in_}")

# Example: Make predictions on test data
print("\n" + "="*50)
print("Making Predictions on Test Data")
print("="*50)

# Load test data
X_test = pd.read_csv('models/x_test.csv')
y_test = pd.read_csv('models/y_test.csv')

# Make predictions
predictions = model.predict(X_test)

# Show first 10 predictions vs actual
print("\nFirst 10 Predictions vs Actual:")
print("-" * 50)
results_df = pd.DataFrame({
    'Actual': y_test.values.flatten()[:10],
    'Predicted': predictions[:10],
    'Difference': y_test.values.flatten()[:10] - predictions[:10]
})
print(results_df.to_string(index=False))

# Show statistics
print("\n" + "="*50)
print("Prediction Statistics")
print("="*50)
print(f"Mean Actual: {y_test.values.mean():.4f}")
print(f"Mean Predicted: {predictions.mean():.4f}")
print(f"Min Prediction: {predictions.min():.4f}")
print(f"Max Prediction: {predictions.max():.4f}")
print(f"Std Dev: {predictions.std():.4f}")

# Load and display metrics
print("\n" + "="*50)
print("Model Performance Metrics")
print("="*50)
with open('models/metrics.txt', 'r') as f:
    print(f.read())

# Example: Make prediction on new data
print("\n" + "="*50)
print("Example: Predict on New Sample")
print("="*50)

# Take first row of test data as example
sample = X_test.iloc[0:1]
sample_prediction = model.predict(sample)[0]
sample_actual = y_test.iloc[0].values[0]

print(f"Sample Input Features:\n{sample.T}")
print(f"\nPredicted 3 Year Return: {sample_prediction:.4f}")
print(f"Actual 3 Year Return: {sample_actual:.4f}")
print(f"Prediction Error: {abs(sample_prediction - sample_actual):.4f}")
