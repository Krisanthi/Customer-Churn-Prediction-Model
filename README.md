
# Telco Customer Churn Prediction Model

<div align="center">

  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
  ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
  ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
  ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
  ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

This repository contains a machine learning pipeline designed to predict telecommunications customer churn. The project utilizes a Decision Tree Classifier and a Deep Neural Network to analyze customer data, identify key churn drivers, and deliver actionable business insights.

## Technical Highlights

* **Advanced Preprocessing:** Executed robust data cleaning, including missing value imputation, Z-score outlier removal, and standard scaling for numerical features.
* **Class Imbalance Handling:** Applied the Synthetic Minority Over-sampling Technique (SMOTE) to rectify a 2.77 to 1 class imbalance, ensuring fair and accurate model training.
* **Neural Network Architecture:** Designed a 4-layer deep learning model (128, 64, 32, and 16 neurons) utilizing ReLU activation, Dropout regularization to prevent overfitting, and the Adam optimizer.
* **Hyperparameter Optimization:** Conducted exhaustive GridSearchCV testing across 270 parameter combinations with 5-fold cross-validation to precisely tune the Decision Tree Classifier.

## System Architecture

The analysis and modeling pipeline is structured into three core phases.

**1. Exploratory Data Analysis (EDA)**
* Generated statistical summaries and correlation matrices, identifying a strong positive correlation (r=0.826) between tenure and total charges.
* Visualized churn distribution across categorical variables using targeted box and violin plots.

**2. Model Training**
* **Decision Tree:** Configured with Gini impurity, a maximum depth of 10, and a minimum samples split of 5.
* **Neural Network:** Trained utilizing early stopping mechanisms (patience of 10) to optimize epochs and prevent validation loss degradation.

**3. Evaluation & Insights**
* **Neural Network Performance:** Achieved 81.06% Accuracy, 85.99% Recall, and an ROC-AUC of 0.8833, significantly outperforming the Decision Tree baseline.
* **Key Findings:** Month-to-month contracts (42.7% churn), fiber optic services (41.9% churn), and electronic check payments (45.3% churn) were identified as the primary drivers of customer attrition.

## Repository Structure

* `Telco_Customer_Churn_Prediction.ipynb`: Primary Jupyter Notebook containing the full EDA, preprocessing, and modeling pipeline.
* `CM2604_Coursework_Report.pdf`: Comprehensive academic report detailing the methodology and statistical findings.

## Local Setup & Execution

**1. Clone the Repository**
```bash
git clone [https://github.com/Krisanthi/Customer-Churn-Prediction-Model.git](https://github.com/Krisanthi/Customer-Churn-Prediction-Model.git)
cd Customer-Churn-Prediction-Model
```

**2. Run in Google Colab**
* Navigate to Google Colab.
* Upload `Telco_Customer_Churn_Prediction.ipynb`.
* Download the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset from Kaggle.
* Run all cells sequentially, uploading the dataset when prompted by the notebook environment.

## AI Ethics & Considerations

* **Data Privacy:** Stripped all personally identifiable information, including CustomerID, prior to processing.
* **Fairness:** Mitigated dataset bias through SMOTE and actively monitored sensitive features for disparate impact.
* **Transparency:** Documented explicit feature importance rules via the Decision Tree model to ensure high business interpretability.
