import pytest
import pandas as pd
import joblib
import yaml
from pathlib import Path

def test_model_exists():
    """Test if model file exists"""
    assert Path("models/best_random_forest_model.pkl").exists()

def test_model_loads():
    """Test if model can be loaded"""
    model = joblib.load("models/best_random_forest_model.pkl")
    assert model is not None

def test_model_predictions():
    """Test if model can make predictions"""
    model = joblib.load("models/best_random_forest_model.pkl")
    X_test = pd.read_csv("models/x_test.csv")
    
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert predictions.dtype in ['float64', 'float32']

def test_metrics_exist():
    """Test if metrics file exists"""
    assert Path("models/metrics.txt").exists()

def test_performance_threshold():
    """Test if model meets minimum performance"""
    with open("models/metrics.txt", "r") as f:
        lines = f.readlines()
    
    r2_line = [l for l in lines if "R2 Score" in l][0]
    r2_score = float(r2_line.split(":")[1].strip())
    
    assert r2_score > 0.5, f"R2 score {r2_score} is below threshold"
