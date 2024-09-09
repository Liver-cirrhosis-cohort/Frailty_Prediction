# model_validation.py

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_predict, cross_val_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv('data/TPHCD_validation dataset.csv')

# Define the feature columns and target variable
features = ['Sex', 'Age', 'BMI', 'CVH', 'Alcohol', 'Autoimmune', 'Cryptogenic', 
             'Ascites', 'HE', 'EGVH', 'Infection', 'Albumin', 'Hb', 'TBIL', 
             'Creatinine', 'INR', 'NLR', 'PLR', 'LMR', 'MELD', 'CTP', 
             'ALT', 'AST', 'ALP', 'GGT', 'LDH', 'PLT', 'Na']
y = df['Frail']
X = df[features]

# Define the models to be evaluated
models = [
    ('Logit', LogisticRegression(max_iter=100, C=0.1)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('ANN', MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', random_state=0)),
    ('RF', RandomForestClassifier(max_depth=2, random_state=0)),
    ('XGBoost', XGBClassifier(max_depth=2, n_estimators=100, learning_rate=0.05, random_state=42))
]

# Evaluate models using cross-validation
for name, model in models:
    auc_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print(f'{name} AUC: {auc_scores.mean():.2f} (+/- {auc_scores.std() * 2:.2f})')

# Plot ROC curves
plt.figure(figsize=(10, 7))
for name, model in models:
    y_probas = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Model Validation')
plt.legend(loc="lower right")
plt.show()