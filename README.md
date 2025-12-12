# Telco Customer Churn Prediction - Python

## Overview
This project focuses on analyzing a telecommunications dataset to predict customer churn using machine learning techniques. It employs a Decision Tree Classifier and a Neural Network to develop robust predictive models for identifying customers likely to discontinue their services.

## Objectives
* Analyze the factors influencing customer churn decisions.
* Build and evaluate machine learning models to predict customer churn.
* Compare model performance to derive actionable business insights.
* Address ethical considerations in AI model development and deployment.

## Dataset
* Source: IBM Telco Customer Churn Dataset (Kaggle)
* Dataset Details:
   * File: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
   * Rows: 7,043
   * Columns: 21
   * Target Variable: `Churn` (binary classification: "Yes" or "No")

## Key Steps

### Data Preprocessing
1. Data Type Conversion: Convert TotalCharges from object to numeric format.
2. Missing Value Imputation: Fill 11 missing values with median ($1,397.47).
3. Feature Removal: Remove CustomerID (non-predictive identifier).
4. Binary Encoding: Convert binary categorical variables (gender, Partner, Dependents, etc.).
5. One-Hot Encoding: Transform multi-category variables (Contract, PaymentMethod, InternetService).
6. Outlier Removal: Use Z-score method (|Z| > 3) to eliminate extreme values.
7. Feature Scaling: Standardize numerical features using StandardScaler.
8. Balancing Classes: Apply SMOTE (k_neighbors=5) to address 2.77:1 class imbalance.

### Exploratory Data Analysis (EDA)
* Statistical summary of numerical and categorical features.
* Correlation heatmap revealing tenure-TotalCharges relationship (r=0.826).
* Churn rate analysis by categorical features.
* Distribution visualizations with box plots and violin plots.

### Model Development
* Decision Tree Classifier:
   * Hyperparameter Tuning: GridSearchCV with 270 parameter combinations
   * Best Parameters: `criterion='gini'`, `max_depth=10`, `min_samples_split=5`
   * 5-Fold Cross-Validation for robust evaluation
* Neural Network Model:
   * Architecture: 4 hidden layers (128→64→32→16 neurons) with ReLU activation
   * Regularization: Dropout layers (0.3, 0.3, 0.2) to prevent overfitting
   * Training: Adam optimizer (lr=0.001), Early Stopping (patience=10)

### Model Evaluation
* Decision Tree Results:
   * Accuracy: 78.50%
   * Precision: 76.20%
   * Recall: 82.90%
   * F1-Score: 79.41%
   * ROC-AUC: 0.8434
   * Feature Importance: Contract type, tenure, and MonthlyCharges dominate.
* Neural Network Results:
   * Accuracy: 81.06%
   * Precision: 78.28%
   * Recall: 85.99%
   * F1-Score: 81.95%
   * ROC-AUC: 0.8833

## Results
* Both models demonstrate strong predictive performance with no significant overfitting.
* The Neural Network outperforms the Decision Tree across all metrics.
* Key churn drivers identified: month-to-month contracts (42.7% churn), fiber optic service (41.9% churn), electronic check payments (45.3% churn).
* Recommended model: Neural Network for deployment based on superior ROC-AUC performance.

## Repository Structure
* `Telco_Customer_Churn_Prediction.ipynb`: Complete analysis and model building.
* `CM2604_Coursework_Report.pdf`: Academic report with detailed methodology.
* `README.md`: Project documentation.

## How to Use
1. Clone the repository:

```
git clone https://github.com/Krisanthi/Customer-Churn-Prediction-Model.git
```

2. Open in Google Colab:

```
Upload Telco_Customer_Churn_Prediction.ipynb to Google Colab
```

3. Upload the dataset when prompted and run all cells sequentially.

## Dependencies
* Python 3.x
* Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, tensorflow, imbalanced-learn, scipy

## AI Ethics Considerations
* Data Privacy: CustomerID removed to protect personally identifiable information.
* Fairness: SMOTE applied for balanced class representation; gender feature monitored for bias.
* Transparency: Decision Tree provides interpretable rules; feature importance documented.
* Post-Deployment: Performance monitoring, bias audits, and human oversight recommended.

## Future Work
* Hyperparameter optimization for Neural Network using GridSearchCV or Bayesian methods.
* Explore ensemble models (Random Forest, XGBoost, LightGBM).
* Implement SHAP values for enhanced model interpretability.
* Develop real-time prediction API using Flask/FastAPI.
* Create interactive dashboard with Streamlit.

## License
This project is licensed under the MIT License.

## Acknowledgments
* Dataset: IBM via Kaggle - Telco Customer Churn Dataset
* Inspiration: Coursework from CM2604 Machine Learning module.
* Special Gratitude to Mr. Sahan Priyanayana and CM2604 Module Team.
