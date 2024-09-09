import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
import random

# Load data
df = pd.read_csv('data/TJMUGH_training&test_dataset.csv')
all_features = ['Sex', 'Age', 'BMI', 'CVH', 'Alcohol', 'Autoimmune', 'Cryptogenic', 'Ascites', 'HE', 'EGVH', 'Infection', 'Albumin', 'Hb', 'TBIL', 'Creatinine', 'INR', 'NLR', 'PLR', 'LMR', 'MELD', 'CTP', 'ALT', 'AST', 'ALP', 'GGT', 'LDH', 'PLT', 'Na']
y = df['Frail']

# Models to train
models = [
    ('Logit', LogisticRegression(max_iter=10000, C=0.1), all_features),
    ('KNN', KNeighborsClassifier(n_neighbors=5), all_features),
    ('ANN', MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=0), all_features),
    ('RF', RandomForestClassifier(max_depth=2, random_state=0), all_features),
    ('XGBoost', XGBClassifier(max_depth=2, n_estimators=100, learning_rate=0.05, random_state=42), all_features)
]

# Set random seed and initialize ROC data storage
n_bootstraps = 500
roc_data = {}
np.random.seed(42)
random.seed(42)

# Train models and collect ROC data
for name, model, features in models:
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    for X_set, y_set, dataset in [(X_train, y_train, 'train'), (X_test, y_test, 'test')]:
        probs = model.predict_proba(X_set)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_set, probs)
        auc_score = auc(fpr, tpr)
        roc_data[(name, dataset, 'AUC')] = auc_score

# Plot ROC curves
plt.figure(figsize=(14, 7))

# Plot ROC for training set
plt.subplot(1, 2, 1)
for name, model, features in models:
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    probs = model.predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison on Training Set')
plt.legend(loc='lower right')

# Plot ROC for test set
plt.subplot(1, 2, 2)
for name, model, features in models:
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison on Test Set')
plt.legend(loc='lower right')

# Show the plots
plt.tight_layout()
plt.show()