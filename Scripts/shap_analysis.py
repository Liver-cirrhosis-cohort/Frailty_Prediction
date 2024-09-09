import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data/TJMUGH_training&test_dataset.csv')

# Define features and target variable
features = ['Sex', 'Age', 'BMI', 'CVH', 'Alcohol', 'Autoimmune', 'Cryptogenic', 'Ascites', 'HE', 'EGVH', 'Infection', 
            'Albumin', 'Hb', 'TBIL', 'Creatinine', 'INR', 'NLR', 'PLR', 'LMR', 'MELD', 'CTP', 'ALT', 'AST', 'ALP', 
            'GGT', 'LDH', 'PLT', 'Na']
y = df['Frail']
X = df[features]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a random forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SHAP analysis
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Bar Plot and Beeswarm Plot
shap.summary_plot(shap_values[1], X_test, plot_type="bar")
shap.summary_plot(shap_values[1], X_test, plot_type="dot")

# Dependence Plot for LMR
feature_index = X_test.columns.get_loc('LMR')
feature_values = X_test['LMR']
shap.dependence_plot('LMR', shap_values[1], X_test, interaction_index=None, show=False)
plt.scatter(feature_values, shap_values[1][:, feature_index], c=feature_values, cmap='coolwarm')
plt.axhline(0, color='red', linestyle='--', label='SHAP Value = 0')
plt.colorbar(label='LMR')
plt.xlabel('LMR')
plt.ylabel('SHAP Value')
plt.show()

# Dependence Plot for NLR
feature_index = X_test.columns.get_loc('NLR')
feature_values = X_test['NLR']
shap.dependence_plot('NLR', shap_values[1], X_test, interaction_index=None, show=False)
plt.scatter(feature_values, shap_values[1][:, feature_index], c=feature_values, cmap='coolwarm')
plt.axhline(0, color='red', linestyle='--', label='SHAP Value = 0')
plt.colorbar(label='NLR')
plt.xlabel('NLR')
plt.ylabel('SHAP Value')
plt.show()

# Dependence Plot for Creatinine
feature_index = X_test.columns.get_loc('Creatinine')
feature_values = X_test['Creatinine']
shap.dependence_plot('Creatinine', shap_values[1], X_test, interaction_index=None, show=False)
plt.scatter(feature_values, shap_values[1][:, feature_index], c=feature_values, cmap='coolwarm')
plt.axhline(0, color='red', linestyle='--', label='SHAP Value = 0')
plt.colorbar(label='Creatinine')
plt.xlabel('Creatinine')
plt.ylabel('SHAP Value')
plt.show()

# Dependence Plot for Ascites
feature_index = X_test.columns.get_loc('Ascites')
feature_values = X_test['Ascites']
shap.dependence_plot('Ascites', shap_values[1], X_test, interaction_index=None, show=False)
plt.scatter(feature_values, shap_values[1][:, feature_index], c=feature_values, cmap='coolwarm')
plt.axhline(0, color='red', linestyle='--', label='SHAP Value = 0')
plt.colorbar(label='Ascites')
plt.xlabel('Ascites')
plt.ylabel('SHAP Value')
plt.show()

# Dependence Plot for Age
feature_index = X_test.columns.get_loc('Age')
feature_values = X_test['Age']
shap.dependence_plot('Age', shap_values[1], X_test, interaction_index=None, show=False)
plt.scatter(feature_values, shap_values[1][:, feature_index], c=feature_values, cmap='coolwarm')
plt.axhline(0, color='red', linestyle='--', label='SHAP Value = 0')
plt.colorbar(label='Age')
plt.xlabel('Age')
plt.ylabel('SHAP Value')
plt.show()

# Dependence Plot for Albumin
feature_index = X_test.columns.get_loc('Albumin')
feature_values = X_test['Albumin']
shap.dependence_plot('Albumin', shap_values[1], X_test, interaction_index=None, show=False)
plt.scatter(feature_values, shap_values[1][:, feature_index], c=feature_values, cmap='coolwarm')
plt.axhline(0, color='red', linestyle='--', label='SHAP Value = 0')
plt.colorbar(label='Albumin')
plt.xlabel('Albumin')
plt.ylabel('SHAP Value')
plt.show()

# Force Plot
shap_values_instance = explainer.shap_values(X_train.iloc[38])
shap.force_plot(explainer.expected_value[1], shap_values_instance[1], X_train.iloc[38], matplotlib=True)

# Model prediction for the specific instance
y_pred_proba = model.predict_proba(X_train.iloc[[38]])
y_pred = model.predict(X_train.iloc[[38]])
print(y_pred_proba[0][1])
print(y_pred[0])

# Decision Plot
explainer = shap.TreeExplainer(model)
expected_value = explainer.expected_value[1]
features = X_train.iloc[range(100)]
shap_values = explainer.shap_values(features)[1]
feature_sample = features.iloc[38]
shap_values_sample = shap_values[38]
shap.decision_plot(expected_value, shap_values_sample, feature_sample)