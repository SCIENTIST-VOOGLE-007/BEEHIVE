from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
from model.preprocess import predict_bankruptcy  # Import prediction function

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Ensure prediction history file exists
PREDICTION_CSV = "static/predictions.csv"
if not os.path.exists(PREDICTION_CSV):
    pd.DataFrame(columns=["Prediction", "Probability", "Risk Level"]).to_csv(PREDICTION_CSV, index=False)


# 🏠 Home Route (File Upload & Manual Entry Form)
@app.route("/")
def home():
    return render_template("index.html")


# 📂 File Upload Route for CSV Input
@app.route("/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"})

        file = request.files["file"]
        if not file.filename.endswith(".csv"):
            return jsonify({"error": "Only CSV files are allowed"})

        df = pd.read_csv(file)

        # Ensure it has valid numerical features
        if df.shape[1] < 5:  # Avoid tiny datasets
            return jsonify({"error": "CSV must have at least 5 features for prediction."})

        # Extract feature names and first row for prediction
        user_features = df.columns.tolist()
        features = df.iloc[0].tolist()

        # Get bankruptcy prediction & insights
        prediction_results = predict_bankruptcy(features, user_features)

        # Save prediction to history
        save_prediction(prediction_results)

        return render_template(
            "result.html",
            result=prediction_results["prediction"],
            probability=prediction_results["probability"],
            risk_level=prediction_results["risk_level"],
            top_factors=prediction_results["top_factors"],
            industry_comparison=prediction_results.get("industry_comparison", None),
            high_risk_trends=prediction_results.get("high_risk_trends", None)
        )

    except Exception as e:
        return jsonify({"error": str(e)})


# 📝 Manual Data Entry Route
@app.route("/manual", methods=["POST"])
def manual_predict():
    try:
        input_data = [float(request.form[feature]) for feature in request.form]
        user_features = list(request.form.keys())

        prediction_results = predict_bankruptcy(input_data, user_features)

        save_prediction(prediction_results)

        return render_template(
            "result.html",
            result=prediction_results["prediction"],
            probability=prediction_results["probability"],
            risk_level=prediction_results["risk_level"],
            top_factors=prediction_results["top_factors"],
            industry_comparison=prediction_results.get("industry_comparison", None),
            high_risk_trends=prediction_results.get("high_risk_trends", None)
        )

    except Exception as e:
        return jsonify({"error": str(e)})


# 📊 API Route for Power BI & External Systems
@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        if "features" not in data or "feature_names" not in data:
            return jsonify({"error": "Missing 'features' or 'feature_names' in request payload"})

        prediction_results = predict_bankruptcy(data["features"], data["feature_names"])

        save_prediction(prediction_results)

        return jsonify(prediction_results)

    except Exception as e:
        return jsonify({"error": str(e)})


# 📥 Export Predictions as CSV for Power BI
@app.route("/export", methods=["GET"])
def export_csv():
    return send_file(PREDICTION_CSV, as_attachment=True)


# 📌 Function to Save Predictions to CSV (Fixes `append()` Deprecation)
def save_prediction(prediction_results):
    df = pd.read_csv(PREDICTION_CSV)
    new_entry = pd.DataFrame([{
        "Prediction": prediction_results["prediction"],
        "Probability": prediction_results["probability"],
        "Risk Level": prediction_results["risk_level"]
    }])
    df = pd.concat([df, new_entry], ignore_index=True)  # Replaces deprecated append()
    df.to_csv(PREDICTION_CSV, index=False)


# 🚀 Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
