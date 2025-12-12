Telco Customer Churn Prediction - Python
Overview
This project focuses on analyzing the Telco Customer Churn dataset to develop and compare two machine learning models—a Decision Tree Classifier and a Neural Network (ANN)—to predict customer attrition. The primary goal is to provide the telecommunications company with an accurate prediction tool and actionable business insights.

Objectives
Conduct in-depth Exploratory Data Analysis (EDA) on key numerical and categorical features.

Address data quality issues, including class imbalance and feature skewness.

Implement hyperparameter tuning (GridSearchCV) to optimize model performance.

Develop robust predictive models and extract feature importance to explain churn drivers.

Dataset
Source: UCI / Kaggle - Telco Customer Churn Dataset

Target Variable: Churn (Binary: Yes/No)

Key Steps and Model Development
1. Data Preprocessing

Data Cleaning: Handled missing values in TotalCharges using median imputation.

Encoding: Categorical variables were converted using One-Hot Encoding (OHE), avoiding the multicollinearity trap.

Balancing Classes: SMOTE was applied to the training data to correct the ≈2.77:1 class imbalance ratio.

Feature Scaling: Numerical features were standardized using StandardScaler for the Neural Network.

2. Model Implementation

Model	Optimization Strategy	Key Hyperparameters
Decision Tree Classifier	GridSearchCV with 5-fold CV, optimizing for F1-Score.	max_depth: 10, min_samples_split: 5
Neural Network Model	Sequential 128→64→32→16→1 funnel architecture with Dropout (0.3) and Early Stopping (patience=10).	Adam Optimizer (LR 0.001), Binary Cross-Entropy loss.
📈 Model Evaluation and Results
Model	Test Accuracy	Test ROC-AUC	Test F1-Score
Decision Tree (Tuned)	78.50%	0.8434	0.7941
Neural Network (ANN)	81.06%	0.8833	0.8195
Key Conclusion

The Neural Network model achieved superior performance (highest ROC-AUC and F1-Score), proving to be the most robust predictor, while the Decision Tree provided the best transparency for rule extraction.

 Feature Insights
Feature importance analysis confirms that economic and commitment factors are the primary drivers of churn:

Top Predictors: tenure, TotalCharges, and MonthlyCharges (the customer's financial profile).

Highest Risk: Customers on Month-to-Month Contracts and those using Fiber Optic internet are the most vulnerable segments.

Lowest Impact: Demographic features like gender and SeniorCitizen have minimal predictive power.

 How to Use
Clone the repository:

Bash
git clone <repository-link>
Open the Notebook: Run all cells in CM2604_Churn_Prediction_Colab-5.ipynb via Google Colab or Jupyter Lab.

Dependencies

Python 3.x, pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, keras, imblearn.

Acknowledgments
Dataset: UCI / Kaggle - Telco Customer Churn Dataset

Inspiration: Coursework from CM2604 Machine Learning module.

References: Comprehensive Harvard-style list provided in the notebook.
