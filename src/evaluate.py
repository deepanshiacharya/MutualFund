import pandas as pd
import joblib
import yaml
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Load config
with open('params.yaml', 'r') as f:
    config = yaml.safe_load(f)

def evaluate_model():
    """Evaluate the saved model"""
    print("Loading model and test data...")
    
    # Load model
    model = joblib.load("models/best_random_forest_model.pkl")
    
    # Load test data
    X_test = pd.read_csv("models/x_test.csv")
    y_test = pd.read_csv("models/y_test.csv")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nEvaluation Metrics:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test.values.flatten(),
        'predicted': y_pred,
        'residual': y_test.values.flatten() - y_pred
    })
    predictions_df.to_csv("models/predictions.csv", index=False)
    print("\nPredictions saved to models/predictions.csv")
    
    return r2, mse, rmse, mae

if __name__ == "__main__":
    evaluate_model()
