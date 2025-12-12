# 🔮 Telco Customer Churn Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Academic](https://img.shields.io/badge/Academic-Coursework-purple.svg)]()

> **A comprehensive machine learning solution for predicting customer churn in telecommunications, implementing Decision Tree and Neural Network classifiers with advanced preprocessing, hyperparameter optimization, and ethical AI considerations.**

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Technical Architecture](#-technical-architecture)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Ethical Considerations](#-ethical-considerations)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 🎯 Project Overview

Customer churn prediction is a critical business challenge in the telecommunications industry, where acquiring new customers costs significantly more than retaining existing ones. This project develops a robust machine learning pipeline to identify customers at high risk of churning, enabling proactive retention strategies.

### Objectives

- Perform comprehensive **Exploratory Data Analysis (EDA)** to uncover churn patterns and drivers
- Implement and compare **Decision Tree** and **Neural Network** classification models
- Apply industry-standard **data preprocessing** techniques including SMOTE for class balancing
- Conduct systematic **hyperparameter optimization** using GridSearchCV
- Evaluate models using multiple metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Address **AI ethics** considerations for responsible model deployment

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Dual-Model Approach** | Comparative analysis of interpretable Decision Trees and high-performance Neural Networks |
| **Comprehensive EDA** | 15+ visualizations including correlation heatmaps, distribution plots, and churn analysis |
| **Advanced Preprocessing** | 7-step pipeline with missing value imputation, encoding, outlier removal, and scaling |
| **Class Imbalance Handling** | SMOTE implementation to address 2.77:1 class imbalance ratio |
| **Hyperparameter Tuning** | GridSearchCV with 270 parameter combinations for Decision Tree optimization |
| **Overfitting Detection** | Training vs. test performance comparison with automated warning system |
| **Ethical AI Framework** | Privacy preservation, fairness considerations, and post-deployment monitoring strategies |

---

## 🏗 Technical Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                             │
│                   Telco Customer Churn Dataset                          │
│                      (7,043 customers × 21 features)                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXPLORATORY DATA ANALYSIS                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │  Statistical │  │   Feature    │  │    Data      │                  │
│  │   Summary    │  │  Correlation │  │ Visualization│                  │
│  └──────────────┘  └──────────────┘  └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING PIPELINE                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │ Type    │ │ Missing │ │Feature  │ │ Outlier │ │ Feature │           │
│  │Conversion│→│ Value   │→│Encoding │→│ Removal │→│ Scaling │           │
│  │         │ │Imputation│ │         │ │(Z-score)│ │(Standard)│          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      CLASS BALANCING (SMOTE)                            │
│              Before: 73.5% / 26.5%  →  After: 50% / 50%                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        MODEL TRAINING                                   │
│  ┌─────────────────────────┐    ┌─────────────────────────┐            │
│  │     DECISION TREE       │    │     NEURAL NETWORK      │            │
│  │  ───────────────────    │    │  ───────────────────    │            │
│  │  • GridSearchCV         │    │  • 4 Hidden Layers      │            │
│  │  • 270 Combinations     │    │  • Dropout Regularization│           │
│  │  • 5-Fold CV            │    │  • Early Stopping       │            │
│  │  • Gini/Entropy Split   │    │  • Adam Optimizer       │            │
│  └─────────────────────────┘    └─────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        MODEL EVALUATION                                 │
│        Accuracy │ Precision │ Recall │ F1-Score │ ROC-AUC              │
│        Confusion Matrix │ ROC Curves │ Feature Importance              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**Source:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)

### Dataset Characteristics

| Attribute | Value |
|-----------|-------|
| **Total Records** | 7,043 customers |
| **Features** | 21 attributes |
| **Numerical Features** | 3 (tenure, MonthlyCharges, TotalCharges) |
| **Categorical Features** | 17 (demographics, services, account info) |
| **Target Variable** | Churn (Binary: Yes/No) |
| **Class Distribution** | 73.5% No Churn / 26.5% Churn |

### Feature Categories

- **Demographics:** Gender, SeniorCitizen, Partner, Dependents
- **Services:** PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV, etc.
- **Account:** Contract, PaymentMethod, PaperlessBilling, MonthlyCharges, TotalCharges, Tenure

---

## 🔬 Methodology

### 1. Exploratory Data Analysis

- Statistical summary of numerical and categorical features
- Missing value analysis (11 missing values in TotalCharges)
- Correlation analysis revealing strong tenure-TotalCharges relationship (r=0.826)
- Churn rate analysis by categorical features
- Distribution visualizations with kernel density estimation

### 2. Data Preprocessing Pipeline

```python
Step 1: Data Type Conversion     → TotalCharges: object → numeric
Step 2: Missing Value Imputation → Median imputation ($1,397.47)
Step 3: Feature Removal          → CustomerID (non-predictive)
Step 4: Binary Encoding          → Gender, Partner, Dependents, etc.
Step 5: One-Hot Encoding         → Contract, PaymentMethod, InternetService
Step 6: Outlier Removal          → Z-score method (|Z| > 3)
Step 7: Feature Scaling          → StandardScaler (mean=0, std=1)
```

### 3. Class Balancing

**SMOTE (Synthetic Minority Oversampling Technique)** applied with k_neighbors=5 to generate synthetic churn instances:

| Metric | Before SMOTE | After SMOTE |
|--------|--------------|-------------|
| No Churn | 5,174 (73.5%) | 5,174 (50%) |
| Churn | 1,869 (26.5%) | 5,174 (50%) |

### 4. Model Implementation

#### Decision Tree Classifier

- **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation
- **Parameter Grid:** criterion, max_depth, min_samples_split, min_samples_leaf, max_features
- **Total Combinations:** 270 configurations evaluated
- **Best Parameters:** criterion='gini', max_depth=10, min_samples_split=5

#### Neural Network (Multi-Layer Perceptron)

```
Architecture:
├── Input Layer (30 neurons)
├── Hidden Layer 1 (128 neurons, ReLU, Dropout 0.3)
├── Hidden Layer 2 (64 neurons, ReLU, Dropout 0.3)
├── Hidden Layer 3 (32 neurons, ReLU, Dropout 0.2)
├── Hidden Layer 4 (16 neurons, ReLU)
└── Output Layer (1 neuron, Sigmoid)

Configuration:
• Optimizer: Adam (lr=0.001)
• Loss: Binary Crossentropy
• Early Stopping: patience=10
• Batch Size: 32
```

---

## 📈 Results

### Performance Comparison

| Metric | Decision Tree | Neural Network | Winner |
|--------|---------------|----------------|--------|
| **Accuracy** | 78.50% | **81.06%** | Neural Network |
| **Precision** | 76.20% | **78.28%** | Neural Network |
| **Recall** | 82.90% | **85.99%** | Neural Network |
| **F1-Score** | 79.41% | **81.95%** | Neural Network |
| **ROC-AUC** | 84.34% | **88.33%** | Neural Network |

### Key Findings

1. **Neural Network outperforms Decision Tree** across all metrics with ~4% improvement in ROC-AUC
2. **Top Churn Predictors:** Contract type, Tenure, MonthlyCharges, TotalCharges
3. **High-Risk Segments:**
   - Month-to-month contracts (42.7% churn rate)
   - Fiber optic customers (41.9% churn rate)
   - Electronic check payment users (45.3% churn rate)
4. **No significant overfitting** detected in either model (gap < 10% between train/test)

### Confusion Matrix Results

| Model | True Negatives | False Positives | False Negatives | True Positives |
|-------|----------------|-----------------|-----------------|----------------|
| Decision Tree | 767 | 268 | 177 | 858 |
| Neural Network | 788 | 247 | 145 | 890 |

---

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Clone Repository

```bash
git clone https://github.com/Krisanthi/Customer-Churn-Prediction-Model.git
cd Customer-Churn-Prediction-Model
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn==1.3.2 imbalanced-learn==0.11.0 tensorflow scipy
```

### Requirements File

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn==1.3.2
imbalanced-learn==0.11.0
tensorflow>=2.10.0
scipy>=1.9.0
```

---

## 🚀 Usage

### Running the Notebook

1. **Google Colab (Recommended):**
   - Upload the notebook to Google Colab
   - Upload the dataset when prompted
   - Run all cells sequentially

2. **Local Jupyter Environment:**
   ```bash
   jupyter notebook Telco_Customer_Churn_Prediction.ipynb
   ```

### Quick Start

```python
# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# Load data
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocess (simplified)
# ... (see notebook for complete pipeline)

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(
    criterion='gini', max_depth=10, 
    min_samples_split=5, random_state=42
)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.4f}")
```

---

## 📁 Project Structure

```
Customer-Churn-Prediction-Model/
│
├── 📓 Telco_Customer_Churn_Prediction.ipynb    # Main Jupyter notebook
├── 📄 README.md                                 # Project documentation
├── 📄 requirements.txt                          # Python dependencies
├── 📄 LICENSE                                   # MIT License
│
├── 📂 data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Dataset
│
├── 📂 reports/
│   └── CM2604_Coursework_Report.pdf            # Academic report
│
├── 📂 figures/
│   ├── fig1_churn_distribution.png
│   ├── fig2_correlation_heatmap.png
│   ├── fig3_numerical_distributions.png
│   ├── fig4_boxplots.png
│   ├── fig5_churn_by_categories.png
│   ├── fig6_violin_plots.png
│   ├── fig7_smote_comparison.png
│   ├── fig8_dt_confusion_matrix.png
│   ├── fig9_dt_feature_importance.png
│   ├── fig10_nn_confusion_matrix.png
│   ├── fig11_nn_training_history.png
│   ├── fig12_roc_comparison.png
│   └── fig13_model_comparison.png
│
└── 📂 models/
    ├── decision_tree_model.pkl                  # Saved Decision Tree
    └── neural_network_model.h5                  # Saved Neural Network
```

---

## 🔧 Technologies Used

<table>
<tr>
<td align="center" width="96">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="48" height="48" alt="Python" />
<br>Python
</td>
<td align="center" width="96">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/tensorflow/tensorflow-original.svg" width="48" height="48" alt="TensorFlow" />
<br>TensorFlow
</td>
<td align="center" width="96">
<img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="48" height="48" alt="scikit-learn" />
<br>scikit-learn
</td>
<td align="center" width="96">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" width="48" height="48" alt="Pandas" />
<br>Pandas
</td>
<td align="center" width="96">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" width="48" height="48" alt="NumPy" />
<br>NumPy
</td>
</tr>
<tr>
<td align="center" width="96">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/matplotlib/matplotlib-original.svg" width="48" height="48" alt="Matplotlib" />
<br>Matplotlib
</td>
<td align="center" width="96">
<img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" width="48" height="48" alt="Seaborn" />
<br>Seaborn
</td>
<td align="center" width="96">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/jupyter/jupyter-original.svg" width="48" height="48" alt="Jupyter" />
<br>Jupyter
</td>
<td align="center" width="96">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/git/git-original.svg" width="48" height="48" alt="Git" />
<br>Git
</td>
<td align="center" width="96">
<img src="https://colab.research.google.com/img/colab_favicon_256px.png" width="48" height="48" alt="Colab" />
<br>Colab
</td>
</tr>
</table>

### Libraries & Frameworks

| Category | Technologies |
|----------|--------------|
| **Data Processing** | Pandas, NumPy, SciPy |
| **Machine Learning** | scikit-learn, imbalanced-learn (SMOTE) |
| **Deep Learning** | TensorFlow, Keras |
| **Visualization** | Matplotlib, Seaborn |
| **Development** | Jupyter Notebook, Google Colab |
| **Version Control** | Git, GitHub |

---

## ⚖️ Ethical Considerations

### Development Phase

- **Data Privacy:** CustomerID removed to prevent re-identification; no PII in training data
- **Fairness:** Gender feature monitored for bias (low importance: 0.012); SMOTE applied for balanced representation
- **Transparency:** Decision Tree provides interpretable rules; feature importance documented

### Post-Deployment Recommendations

- **Performance Monitoring:** Track metrics monthly with 5% degradation alerts
- **Bias Audits:** Quarterly fairness checks across demographic segments
- **Human Oversight:** High-confidence predictions (>85%) reviewed by retention specialists
- **Model Governance:** Version control, A/B testing, and regular retraining schedules

---

## 🔮 Future Enhancements

- [ ] Implement ensemble methods (Random Forest, XGBoost, LightGBM)
- [ ] Add SHAP values for enhanced model interpretability
- [ ] Develop real-time prediction API using Flask/FastAPI
- [ ] Create interactive dashboard with Streamlit
- [ ] Implement automated retraining pipeline
- [ ] Add k-fold cross-validation for robust evaluation
- [ ] Explore feature engineering (interaction terms, polynomial features)
- [ ] Conduct threshold optimization for business-specific cost functions

---

## 👤 Author

**Krisanthi Segar**

- 🎓 BSc (Hons) Artificial Intelligence and Data Science
- 🏫 Robert Gordon University Aberdeen / Informatics Institute of Technology
- 📧 Email: [your-email@example.com]
- 💼 LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
- 🐙 GitHub: [@Krisanthi](https://github.com/Krisanthi)

### Academic Information

| Field | Details |
|-------|---------|
| **Module** | CM2604 Machine Learning |
| **Programme** | BSc (Hons) AI and Data Science |
| **RGU Student ID** | 2425596 |
| **IIT Student ID** | 20232384 |
| **Supervisor** | Mr. Sahan Priyanayana |

---

## 🙏 Acknowledgements

- **IBM** for providing the Telco Customer Churn dataset
- **Kaggle** for hosting the dataset and community resources
- **Mr. Sahan Priyanayana** for module coordination and guidance
- **Robert Gordon University** and **Informatics Institute of Technology** for academic support
- Open-source community for the amazing libraries and tools

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Krisanthi Segar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

[![GitHub stars](https://img.shields.io/github/stars/Krisanthi/Customer-Churn-Prediction-Model?style=social)](https://github.com/Krisanthi/Customer-Churn-Prediction-Model)

**Made with ❤️ for Machine Learning**

</div>
