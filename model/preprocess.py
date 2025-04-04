import pickle
import numpy as np
import pandas as pd
import os

# File Paths
SCALER_PATH = "model/scaler.pkl"
MODEL_PATH = "model/bankruptcy_model.pkl"
FEATURE_IMPORTANCE_PATH = "model/feature_importance.csv"
FEATURES_PATH = "model/features.pkl"

# Ensure required files exist
for path in [SCALER_PATH, MODEL_PATH, FEATURE_IMPORTANCE_PATH, FEATURES_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

# Load model & scaler
with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)
with open(FEATURES_PATH, "rb") as feature_file:
    trained_features = pickle.load(feature_file)  # Load features from training

# Load feature importance
df_feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)

def preprocess_input(data, input_features):
    """
    Dynamically aligns user input to the trained model's feature set.
    Args:
        data (list): A list of numerical financial values.
        input_features (list): Feature names from user input.
    Returns:
        np.array: Standardized input data for the model.
    """
    # Ensure dataset structure matches training
    input_df = pd.DataFrame([data], columns=input_features)

    # Align with trained feature set (fill missing, drop extra)
    for feature in trained_features:
        if feature not in input_df:
            input_df[feature] = 0  # Fill missing features with 0
    input_df = input_df[trained_features]  # Ensure correct order

    # Standardize the data
    processed_data = scaler.transform(input_df)

    return processed_data

def predict_bankruptcy(features, feature_names):
    """
    Predicts bankruptcy risk and provides insights.
    Args:
        features (list): Numerical input features.
        feature_names (list): Column names of the dataset.
    Returns:
        dict: Prediction results including probability, risk category, and insights.
    """
    # Preprocess user input
    input_data = preprocess_input(features, feature_names)

    # Get prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100  # Convert to percentage

    # Risk Segmentation
    risk_level = "High Risk" if probability > 70 else "Medium Risk" if probability > 40 else "Low Risk"

    # Top 5 Most Important Factors
    top_factors = df_feature_importance.sort_values(by="Importance", ascending=False).head(5).to_dict(orient="records")

    return {
        "prediction": "Bankrupt" if prediction == 1 else "Not Bankrupt",
        "probability": round(probability, 2),
        "risk_level": risk_level,
        "top_factors": top_factors
    }
