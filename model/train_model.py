import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# 📂 **Load dataset**
df = pd.read_csv("data.csv")

# 🛠 **Step 1: Clean Column Names (Remove Spaces & Special Characters)**
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('?', '')

# 🔍 **Step 2: Detect Target Column Automatically**
possible_targets = ["Bankrupt", "Bankrupt_", "status", "class", "Default", "Failure", "Risk_Level"]
target = next((col for col in df.columns if col in possible_targets), None)

if not target:
    raise ValueError("❌ No valid target column found in dataset! Please check column names.")

print(f"🎯 Target column detected: {target}")

# 📊 **Step 3: Select Only Numerical Features (Ignore Text Columns)**
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

# ❌ **Remove Target Column from Features**
if target in numerical_features:
    numerical_features.remove(target)

if len(numerical_features) == 0:
    raise ValueError("❌ No numerical features found in dataset! Please check the data.")

print(f"📌 Selected {len(numerical_features)} numerical features for training.")

# 🏗 **Step 4: Handle Missing Values**
df.fillna(df[numerical_features].median(), inplace=True)

# 🏗 **Step 5: Define X (features) and y (target)**
X = df[numerical_features].values
y = df[target].values

# ✂ **Step 6: Split Dataset**
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔬 **Step 7: Standardize Features**
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🤖 **Step 8: Train XGBoost Model**
classifier_xg = XGBClassifier(max_depth=4, n_estimators=200, random_state=42)
classifier_xg.fit(X_train, y_train)

# 📈 **Step 9: Evaluate Model**
y_pred = classifier_xg.predict(X_test)
y_proba = classifier_xg.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, y_proba)
print(f'✅ ROC AUC Score: {roc_score:.4f}')
print(classification_report(y_test, y_pred))

# 🔥 **Step 10: Save Feature Importance**
feature_importance = classifier_xg.feature_importances_
importance_df = pd.DataFrame({"Feature": numerical_features, "Importance": feature_importance})
importance_df.sort_values(by="Importance", ascending=False, inplace=True)
importance_df.to_csv("model/feature_importance.csv", index=False)

# 📊 **Step 11: Industry & Financial Trends Analysis**
industry_metric, revenue_col, expense_col = None, None, None

# ✅ **Detect relevant financial columns dynamically**
for col in df.columns:
    if "Asset_Turnover" in col:
        industry_metric = col
    if "Revenue" in col:
        revenue_col = col
    if "Expense" in col:
        expense_col = col

# 📈 **Industry Analysis**
if industry_metric and revenue_col:
    df_industry = df.groupby(pd.qcut(df[industry_metric], q=4, duplicates="drop"))[
        [revenue_col, "Net_Profit"] if "Net_Profit" in df.columns else [revenue_col]
    ].mean()
    df_industry.to_csv("model/industry_health.csv")
    print("✅ Industry analysis saved!")
else:
    print("⚠️ No industry segmentation column found. Skipping industry analysis.")

# 📉 **Expense vs Revenue Trends**
if revenue_col and expense_col:
    high_risk_companies = df[df[target] == 1][[revenue_col, expense_col]]
    high_risk_companies.to_csv("model/high_risk_expense_trends.csv", index=False)
    print("✅ High-risk financial trends saved!")
else:
    print("⚠️ No revenue/expense columns found. Skipping high-risk company trends.")

fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_score:.4f}", color='orange', linewidth=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Save the ROC Curve to the static folder
roc_path = "static/roc_curve.png"

# 🔥 Ensure the plot is properly saved and closed
plt.savefig(roc_path, bbox_inches='tight', dpi=300)  # Save with high quality
plt.close()  # Close the figure to free memory

print(f"✅ ROC curve saved successfully at {roc_path}")

# 📂 **Step 13: Save Model & Scaler**
os.makedirs("model", exist_ok=True)
with open("model/bankruptcy_model.pkl", "wb") as model_file:
    pickle.dump(classifier_xg, model_file)
with open("model/scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)
with open("model/features.pkl", "wb") as feature_file:
    pickle.dump(numerical_features, feature_file)

print("✅ Model, feature importance, and financial insights saved successfully!")
