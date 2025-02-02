# Frailty Prediction Model Using Machine Learning
This repository contains code for building and evaluating machine learning models for frailty prediction based on a variety of clinical features.

## Project Structure
- **data/**: This folder contains the training and test datasets used for model building and validation.
- **scripts/**: Python scripts for model training, evaluation, and SHAP-based feature importance analysis.
- **README.md**: Project documentation.
- **requirements.txt**: List of Python dependencies required to run the project.

## Data
- `TJMUGH_training&test_dataset.csv`: The training and test dataset used for model training and evaluation.
- `TPHCD_validation_dataset.csv`: The validation dataset used to test the model's performance on new data.

## Scripts
- `model_training_and_evaluation.py`: This script trains various machine learning models, evaluates them using ROC curves, and plots the results.
- `shap_analysis.py`: This script performs SHAP analysis to evaluate feature importance and create dependence plots.
- `model_validation.py`: This script evaluates various machine learning models on the validation dataset, computes AUC scores using cross-validation, and plots ROC curves.

## Setup and Installation

1. Clone the repository:
   ```bash
git clone https://github.com/Liver-cirrhosis-cohort/Frailty_Prediction.git
   cd project_name
2. Install the required dependencies:
pip install -r requirements.txt
3. Run the model training and evaluation script:
python scripts/model_training_and_evaluation.py
4. Run SHAP analysis:
python scripts/shap_analysis.py
5. Run the model validation script:
python scripts/model_validation.py

