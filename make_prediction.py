import joblib
import pickle
import numpy as np
import pandas as pd

# Load model and preprocessors
model = joblib.load('models/best_random_forest_model.pkl')

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_return(features_dict):
    """
    Make a prediction on new data
    
    Args:
        features_dict: Dictionary with feature names and values
    
    Returns:
        Predicted 3-year return
    """
    # Convert to DataFrame
    df = pd.DataFrame([features_dict])
    
    # Scale features
    scaled_features = scaler.transform(df)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    
    return prediction

if __name__ == "__main__":
    # Example usage - replace with your actual feature names and values
    # Get feature names from test data
    X_test = pd.read_csv('models/x_test.csv')
    feature_names = X_test.columns.tolist()
    
    print("Available features:", feature_names)
    print("\nMaking sample predictions...")
    
    # Use a sample from test data
    for i in range(3):
        sample = X_test.iloc[i].to_dict()
        predicted_return = predict_return(sample)
        print(f"\nSample {i+1}:")
        print(f"Features: {sample}")
        print(f"Predicted 3 Year Return: {predicted_return:.4f}")
