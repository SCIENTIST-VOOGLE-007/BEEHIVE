# BEEHIVE: Corporate Health Predictor 🐝

> Predicting the financial future—because even corporations need a health check.Predict and visualize a company’s financial health using Flask, XGBoost, and Power BI | Corporate Bankruptcy Risk Predictor

## 🧠 Overview

BEEHIVE is a Flask-based web application designed to predict the financial health of companies using Machine Learning. By analyzing critical financial metrics, BEEHIVE forecasts bankruptcy risks and provides data-driven insights, helping stakeholders make smarter decisions.

## 🎯 Features

- 📊 Predicts bankruptcy risk with 88% accuracy using XGBoost.
- 📉 Analyzes over 10+ financial metrics including:
  - Liquidity ratios
  - Profitability margins
  - Debt levels
  - Solvency metrics
- 📈 Interactive Power BI dashboards for clear visual interpretation.
- ⚙️ Seamless web integration via Flask.

## 🛠️ Tech Stack

- **Frontend**: HTML/CSS (Basic Flask templating)
- **Backend**: Python, Flask
- **ML Model**: XGBoost, Scikit-learn
- **Visualization**: Power BI

## 📦 Installation

```bash
git clone https://github.com/your-username/BEEHIVE.git
cd BEEHIVE
pip install -r requirements.txt
```

## Usage
Run the Flask server:

```bash
python app.py
```
Upload company financial data (CSV format).

View prediction and risk status.

Access the Power BI dashboard for detailed insights.

📊 Sample Inputs
csv
Copy
Edit
Company,Current Ratio,Debt Ratio,Net Profit Margin,...
XYZ Corp,1.8,0.45,0.12,...
🔍 Model
The XGBoost classifier was trained on a curated dataset from over 10 companies. The features were engineered using domain knowledge, and the model was validated using 5-fold cross-validation.

🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

