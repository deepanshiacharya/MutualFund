import pandas as pd
import numpy as np
import pickle
import yaml
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

# Load config
with open('params.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set MLflow tracking
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

def load_and_preprocess_data():
    """Load and preprocess the data"""
    print("Loading data...")
    df = pd.read_csv(config['data']['raw_data_path'])
    
    # Label encode categorical features
    le = LabelEncoder()
    label_encoders = {}
    
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    
    # Save label encoders
    Path("models").mkdir(exist_ok=True)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Split features and target
    X = df.drop(config['data']['target_column'], axis=1)
    y = df[config['data']['target_column']]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save test data
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('models/x_test.csv', index=False)
    pd.DataFrame(y_test).to_csv('models/y_test.csv', index=False)
    
    print(f"Data preprocessed. Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    """Train model with MLflow tracking"""
    
    with mlflow.start_run():
        
        # Log parameters
        mlflow.log_param("test_size", config['data']['test_size'])
        mlflow.log_param("random_state", config['data']['random_state'])
        
        # Create parameter grid
        param_grid = {
            'n_estimators': config['model']['n_estimators'],
            'max_depth': config['model']['max_depth'],
            'min_samples_split': config['model']['min_samples_split'],
            'min_samples_leaf': config['model']['min_samples_leaf']
        }
        
        # Grid search
        print("Starting GridSearchCV...")
        rf_model = RandomForestRegressor(random_state=config['data']['random_state'])
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=config['model']['cv_folds'],
            n_jobs=-1,
            verbose=2,
            scoring='r2'
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Best parameters: {best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Log best parameters
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nTest Metrics:")
        print(f"R2 Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # Log metrics
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Save model locally
        joblib.dump(best_model, "models/best_random_forest_model.pkl")
        print("Model saved to models/best_random_forest_model.pkl")
        
        # Save metrics to file
        with open("models/metrics.txt", "w") as f:
            f.write(f"R2 Score: {r2:.4f}\n")
            f.write(f"MSE: {mse:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
        
        return best_model, r2, mse, rmse, mae

if __name__ == "__main__":
    # Preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Train model
    model, r2, mse, rmse, mae = train_model(X_train, X_test, y_train, y_test)
    
    print("\nTraining complete!")
