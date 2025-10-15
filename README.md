# Mutual Fund Return Prediction - MLOps Project

##  Project Description

This project demonstrates a complete **Machine Learning Operations (MLOps)** implementation for predicting 3-year returns of mutual funds using historical financial data. The project showcases industry-standard practices for building, tracking, testing, and deploying machine learning models in a production-ready environment.


##  Project Objectives

The primary goal of this project is to build an automated pipeline that:

1. **Trains** a Random Forest regression model to predict mutual fund returns
2. **Tracks** all experiments, parameters, and metrics using MLflow
3. **Tests** the model automatically with pytest
4. **Deploys** the model through CI/CD pipelines using GitHub Actions
5. **Ensures** reproducibility and version control of the entire ML workflow

##  Problem Statement

**Business Problem**: Investors need accurate predictions of mutual fund performance to make informed investment decisions. This project builds a machine learning model to predict 3-year returns based on historical fund characteristics.

**Technical Challenge**: Implement a complete MLOps solution that includes:
- Experiment tracking and model versioning
- Automated testing and validation
- Continuous Integration and Continuous Deployment (CI/CD)
- Model performance monitoring
- Reproducible and scalable ML workflows

##  What This Project Does

### 1. **Data Processing & Feature Engineering**
- Loads mutual fund dataset with various financial metrics
- Applies label encoding to categorical features
- Performs train-test split (80-20)
- Scales numerical features using StandardScaler
- Saves preprocessing artifacts for consistent inference

### 2. **Model Training with Experiment Tracking**
- Implements Random Forest Regressor with hyperparameter tuning
- Uses GridSearchCV for optimal parameter selection
- Evaluates models using multiple metrics (R2, MSE, RMSE, MAE)
- **Logs everything to MLflow**: parameters, metrics, models, and artifacts
- Tracks experiment history for model comparison and selection

### 3. **Automated Testing**
- Unit tests for model loading and predictions
- Performance threshold validation (R2 > 0.5)
- Automated test execution in CI pipeline
- Code quality checks with flake8

### 4. **CI/CD Automation with GitHub Actions**
Three automated workflows:

**CI Pipeline**: Runs on every commit
- Installs dependencies
- Checks code quality
- Runs test suite
- Validates project structure

**Training Pipeline**: Triggers on code/data changes
- Preprocesses data
- Trains model with hyperparameter tuning
- Evaluates performance
- Uploads model artifacts
- Generates training reports

**Deployment Pipeline**: Packages for production
- Downloads trained model
- Creates deployment package
- Runs pre-deployment tests
- Generates deployment manifest
- Makes artifacts available for download

### 5. **Model Registry & Version Control**
- MLflow model registry tracks all trained models
- Git tracks code versions
- Artifact versioning for reproducibility
- Complete audit trail of all experiments

##  Key Features

 **Experiment Tracking**: Complete visibility into all training runs with MLflow  
 **Automated CI/CD**: GitHub Actions pipelines for testing and deployment  
 **Model Versioning**: Track and compare different model versions  
 **Reproducibility**: Fixed seeds, versioned dependencies, logged artifacts  
 **Performance Monitoring**: Track metrics across training runs  
 **Configuration Management**: Centralized parameters in YAML files  
 **Documentation**: Comprehensive project documentation

## Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Framework** | Scikit-learn | Model training and evaluation |
| **Experiment Tracking** | MLflow | Track experiments, parameters, metrics |
| **CI/CD** | GitHub Actions | Automated testing and deployment |
| **Testing** | Pytest | Unit and integration tests |
| **Version Control** | Git | Code versioning |
| **Configuration** | YAML | Parameter management |
| **Language** | Python 3.10+ | Primary development language |

## Learning Outcomes

This project demonstrates:

1. **MLOps Fundamentals**: Understanding the complete ML lifecycle
2. **Experiment Management**: Tracking and comparing model experiments
3. **Automation**: Building CI/CD pipelines for ML projects
4. **Best Practices**: Industry-standard coding and deployment practices
5. **Reproducibility**: Ensuring consistent results across environments
6. **Version Control**: Managing code, data, and model versions
7. **Testing**: Implementing comprehensive test suites for ML code


