# 🏦 Bank Marketing Classification System

A comprehensive machine learning classification system that predicts whether a client will subscribe to a bank term deposit using 6 different algorithms with hyperparameter optimization and class imbalance handling.

**Assignment:** ML-Assignment2 | M.Tech AI & ML (WILP), Semester 1

---

## 📋 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset Description](#dataset-description)
3. [Models Used](#models-used)
4. [Comparison Table: Model Performance Metrics](#comparison-table-model-performance-metrics)
5. [Model Performance Observations](#model-performance-observations)
6. [Quick Start](#quick-start)
7. [Additional Information](#additional-information)
8. [Project Structure](#project-structure)
9. [Troubleshooting](#troubleshooting)
10. [Key Findings](#key-findings)

---

## 🎯 Problem Statement

This project aims to build a machine learning classification system to predict whether a bank customer will subscribe to a term deposit. The problem involves:

### Objective
Predict whether a bank customer will subscribe to a term deposit based on their demographic and behavioral characteristics. This is a **binary classification problem** with significant class imbalance.

### Business Context
- **Target:** Customers who subscribe to term deposits (minority class: ~11%)
- **Non-Target:** Customers who don't subscribe (majority class: ~89%)
- **Challenge:** Imbalanced dataset requires careful handling to avoid bias towards majority class
- **Goal:** Develop robust classification models that can accurately identify potential customers while maintaining a balance between precision and recall

### Solution Approach
We implemented **6 different machine learning algorithms** with hyperparameter optimization and class imbalance handling techniques to address this challenge and compare their performance on the bank marketing dataset.

---

## 📊 Dataset Description

**Dataset:** Bank Marketing Dataset from UCI Machine Learning Repository  
**Source:** https://archive.ics.uci.edu/dataset/222/bank+marketing

### Dataset Overview
The Bank Marketing dataset contains information about direct marketing campaigns conducted by a Portuguese banking institution. The classification goal is to predict if a client will subscribe to a term deposit.

### Dataset Statistics
- **Total Samples:** 4,119 instances
- **Features:** 20 input features (demographic, economic, and campaign-related attributes)
- **Target:** Binary class 'y' (0: No, 1: Yes)
- **Class Distribution:**
  - Class 0 (No): 3,668 samples (89.1%)
  - Class 1 (Yes): 451 samples (10.9%)
  - **Imbalance Ratio:** 8.13:1

### Feature Breakdown
The 20 features include:
- **Demographic:** age, job, marital status, education, credit default status, housing loan, personal loan
- **Campaign:** contact type (cellular/telephone), day of week, month, duration of last contact, number of contacts during campaign, days since last contact, outcome of previous campaign
- **Economic:** employment variation rate, consumer price index, consumer confidence index, euribor 3-month rate, number of employees

### Data Characteristics
- Missing values handled by removal
- Categorical features encoded using LabelEncoder
- Features scaled using StandardScaler (z-score normalization)
- Train-Test Split: 80% training (3,295 samples), 20% testing (824 samples) with stratification
- Data leakage prevention: Scaler and encoders fitted on training data only

---

## 🤖 Models Used

This project implements and evaluates **6 machine learning algorithms** to compare their performance on the binary classification task:

### 1. **Logistic Regression**
A linear classification model with probabilistic outputs. Used as a baseline model due to its interpretability and efficiency. Class imbalance handled using balanced class weights.

### 2. **Decision Tree**
A tree-based classifier that recursively splits features to maximize information gain. Captures non-linear relationships and is highly interpretable. Class imbalance addressed through balanced class weights and depth control.

### 3. **k-Nearest Neighbors (kNN)**
An instance-based, non-parametric algorithm that classifies samples based on the majority class of k nearest neighbors in the feature space. Distance-based weighting used to handle class imbalance.

### 4. **Naive Bayes**
A probabilistic classifier based on Bayes' theorem with feature independence assumption. Fast and effective for binary classification. SMOTE oversampling applied to handle class imbalance.

### 5. **Random Forest (Ensemble)**
An ensemble method that combines multiple decision trees with bootstrap aggregation. Reduces overfitting and handles non-linearity effectively. Balanced subsample weighting applied.

### 6. **XGBoost (Ensemble)**
An advanced gradient boosting framework that sequentially builds trees to correct previous errors. State-of-the-art performance for structured data. Scale pos weight tuning used for class imbalance.

---

## 📈 Comparison Table: Model Performance Metrics

### All Models - Test Set Performance

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.8993 | 0.8411 | 0.5267 | 0.7667 | 0.6244 | 0.5819 |
| **Decision Tree** | 0.8993 | 0.8265 | 0.5280 | 0.7333 | 0.6140 | 0.5677 |
| **kNN** | 0.8932 | 0.5988 | 0.5263 | 0.2222 | 0.3125 | 0.2940 |
| **Naive Bayes** | 0.8289 | 0.7431 | 0.3455 | 0.6333 | 0.4471 | 0.3790 |
| **Random Forest (Ensemble)** | 0.8993 | 0.8314 | 0.5276 | 0.7444 | 0.6175 | 0.5725 |
| **XGBoost (Ensemble)** | 0.9029 | 0.8188 | 0.5424 | 0.7111 | 0.6154 | 0.5677 |

**Note:** All models were optimized using GridSearchCV with F1-score as the optimization metric, and class imbalance was handled using balanced class weights, SMOTE, or scale_pos_weight depending on the algorithm.

---

## 💡 Model Performance Observations

### Performance Analysis by Model

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieved strong performance with 89.93% accuracy and best-in-class recall (0.7667), indicating it successfully identifies most positive cases. However, precision of 0.5267 suggests approximately 47% of predicted positives are false positives, requiring careful threshold tuning for production use. The AUC of 0.8411 demonstrates good discrimination ability. This model serves as a reliable baseline but trades precision for recall, which is beneficial for minimizing missed customer opportunities in marketing campaigns. |
| **Decision Tree** | Decision Tree achieves balanced performance across metrics with 89.93% accuracy, competitive precision (0.5280), and strong recall (0.7333). The AUC of 0.8265 indicates robust classification ability. The model's interpretability is a major advantage - feature splits can be easily understood and explained to stakeholders. However, the slight precision-recall trade-off and lower AUC compared to some ensemble methods suggest careful pruning is needed to prevent overfitting on test data. Despite these considerations, this model offers excellent explainability for business decisions. |
| **kNN** | kNN surprisingly underperforms with severely limited recall (0.2222), meaning it misses approximately 78% of actual positive cases, making it unsuitable for this marketing task. While precision is reasonable (0.5263), the very low F1 score (0.3125) and poor AUC (0.5988, near random performance) indicate the model struggles with this dataset. This poor performance may be due to high-dimensional feature space, non-uniform feature scaling effects, or inappropriate k-value selection despite GridSearchCV tuning. Recommendation: kNN is not recommended for this problem without significant feature engineering and nearest neighbor approach redesign. |
| **Naive Bayes** | Naive Bayes shows the lowest overall accuracy (0.8289) with relatively poor precision (0.3455), indicating high false positive rates. However, it compensates with strong recall (0.6333), catching roughly 63% of positive cases. The AUC of 0.7431 suggests moderate discriminative ability. The poor performance likely stems from violation of the independence assumption between features (economic indicators are typically correlated). Despite limitations, it remains computationally efficient and the SMOTE preprocessing (60% sampling strategy) helps balance class representation. Performance could potentially improve with feature selection to reduce dependencies. |
| **Random Forest (Ensemble)** | Random Forest delivers excellent performance with 89.93% accuracy, strong precision (0.5276), and highest recall among comparable models (0.7444). The AUC of 0.8314 indicates reliable classification. This ensemble method effectively combines multiple decision trees to reduce variance and capture complex patterns. The balanced subsample weighting handles class imbalance well. The model generalizes efficiently while maintaining interpretability through feature importance scores. However, it's slightly outperformed by XGBoost in accuracy. Overall, Random Forest is a robust, production-ready choice balancing performance, stability, and explainability. |
| **XGBoost (Ensemble)** | XGBoost achieves the **best accuracy (0.9029)** among all models with highest precision (0.5424), demonstrating the fewest false positives (~46% false positive rate). Strong recall (0.7111) ensures most positive cases are captured. Grid search optimization of scale_pos_weight parameter effectively addresses class imbalance. The AUC of 0.8188 represents solid discrimination capability. As a gradient boosting ensemble, XGBoost sequentially corrects errors from previous trees, achieving state-of-the-art performance. Primary limitation: reduced interpretability compared to tree-based alternatives. Recommended as top-choice model for maximum accuracy and precision. |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Download Dataset

1. Go to: https://archive.ics.uci.edu/dataset/222/bank+marketing
2. Download **bank-additional.csv**
3. Place in `data/` folder

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train All Models

```bash
python train_models.py
```

**Output:**
- ✅ All 6 models trained with GridSearchCV and hyperparameter optimization
- ✅ Models saved to `saved_models/`
- ✅ Test data generated: `data/test_data.csv`

### Step 4: Launch Streamlit App

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

### Step 5: Evaluate Models

1. Click "📥 Download Test Data" in sidebar
2. Upload the downloaded CSV file
3. Select a model from dropdown
4. View 6 metrics and visualizations

---

## 📚 Additional Information

### Class Imbalance Handling

#### Problem
- Models tend to predict majority class to maximize accuracy
- Result: High accuracy but poor precision/recall for minority class
- **Original Metrics (Imbalanced):** Precision=0.46, Recall=0.87, F1=0.60

#### Solutions Implemented

**1. Balanced Class Weights**
- Applied to: Logistic Regression, Decision Tree, Random Forest
- Penalizes misclassification of minority class during training
- Example: `{0: 1, 1: 5}` weights minority class 5x more

**2. SMOTE (Synthetic Minority Over-sampling)**
- Applied to: Naive Bayes
- Creates synthetic samples of minority class
- Parameters: `sampling_strategy=0.6, k_neighbors=3`

**3. Scale Pos Weight (XGBoost)**
- XGBoost-specific parameter
- Penalizes false negatives proportionally to class imbalance
- Example: `scale_pos_weight = 8.13 × 2 = 16.26`

#### Results
- ✅ Precision improved by **+15-27%** across models
- ✅ F1 Score improved by **+4-11%** across models
- ✅ Better balanced predictions (not just accuracy)

### Hyperparameter Tuning

**Optimization Strategy:**
- **Method:** GridSearchCV with 5-fold Cross-Validation
- **Scoring Metric:** F1 Score (balances precision & recall)
- **Optimization Goal:** Balanced performance, not just accuracy

**Logistic Regression:**
- C (Regularization): [0.001, 0.01, 0.1, 1, 10]
- Class Weight: ['balanced', {0:1, 1:5}, {0:1, 1:8}]

**Decision Tree:**
- Max Depth: [8, 10, 12, 15]
- Min Samples Split: [5, 10, 15]
- Min Samples Leaf: [2, 4, 5]
- Class Weight: ['balanced', custom weights]

**Random Forest:**
- N Estimators: [100, 150, 200]
- Max Depth: [12, 15, 18, 20]
- Min Samples Split: [8, 10, 12]
- Class Weight: ['balanced', 'balanced_subsample']

**KNN:**
- N Neighbors: [5, 7, 9, 11, 13]
- Weights: ['uniform', 'distance']
- Metric: ['minkowski', 'euclidean', 'manhattan']

**XGBoost:**
- Max Depth: [6, 7]
- Learning Rate: [0.1, 0.15]
- Scale Pos Weight: [8.13×1.5, 8.13×2]

### Evaluation Metrics Explanation

**1. Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
- Overall percentage of correct predictions

**2. Precision:** $\frac{TP}{TP + FP}$
- Of predicted positives, how many were correct?

**3. Recall (Sensitivity):** $\frac{TP}{TP + FN}$
- Of actual positives, how many were found?

**4. F1 Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- Harmonic mean of precision & recall (balances both metrics)

**5. AUC Score:** Area under ROC curve
- Probability model ranks random positive higher than random negative
- Robust to class imbalance

**6. MCC (Matthews Correlation Coefficient):** $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$
- Single score for imbalanced classification (-1 to 1)

---

## 📁 Project Structure

```
ML-Assignment2/
├── README.md                 # Project documentation (this file)
├── INSTRUCTIONS.md          # Quick start guide
├── app.py                   # Streamlit UI application
├── train_models.py          # Training script with GridSearchCV
├── preprocessing.py         # Data preprocessing utilities
├── requirements.txt         # Python dependencies
│
├── models/                  # Model training scripts (reference)
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── knn.py
│   ├── naive_bayes.py
│   └── xgboost_model.py
│
├── saved_models/            # Trained model files (auto-created)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── xgboost.pkl
│
└── data/                    # Dataset files
    ├── bank-additional.csv  # Original dataset (download required)
    └── test_data.csv        # Generated test set (auto-created)
```

---

## 🔧 Troubleshooting

| Error | Solution |
|-------|----------|
| `FileNotFoundError: bank-additional.csv` | Download from UCI repository and place in `data/` folder |
| `ModuleNotFoundError: streamlit` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: saved_models/*.pkl` | Run `python train_models.py` to train models |
| `ValueError: Expected 20 features` | Delete `data/test_data.csv` and retrain with `python train_models.py` |
| `pdftotext not available` | Install `pdftotext` with `pip install pdf2image pdfplumber` |

---

## 💡 Key Findings

### 1. **Class Imbalance is Critical**
- Initial versions with equal class weights resulted in poor precision (0.46)
- Implementing balanced weights, SMOTE, and scale_pos_weight improved precision by 15-27%
- Optimizing for F1 instead of accuracy provides better real-world performance

### 2. **Hyperparameter Tuning Works**
- GridSearchCV optimizing for F1 (not accuracy) improved:
  - F1 Score: +7.47% across models
  - MCC: +5.05% (better balanced metrics)
  - Precision: +5.59%

### 3. **Ensemble Methods Excel**
- Random Forest consistently in top 3
- XGBoost achieves best accuracy (0.9029)
- Decision Tree best precision (0.5280)
- No single "best" model - depends on business goal

### 4. **Trade-offs Exist**
- Logistic Regression: Highest recall (0.7667) but moderate precision
- XGBoost: Highest accuracy (0.9029) and precision (0.5424)
- kNN: Poor performance due to high-dimensional space challenges

### 5. **Evaluation Metric Choice Matters**
- Accuracy alone is misleading (0.90 means little with imbalanced data)
- F1 + Precision + MCC provide balanced view
- AUC is robust to class imbalance

---

## 📖 References

1. **Dataset:** [Bank Marketing Dataset - UCI ML Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
2. **Class Imbalance:** [Imbalanced Learn - SMOTE](https://imbalanced-learn.org/)
3. **Hyperparameter Tuning:** [Scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html)
4. **XGBoost:** [XGBoost Documentation](https://xgboost.readthedocs.io/)
5. **Metrics:** [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**Assignment Status:** ✅ Complete  
**Last Updated:** 13 February 2026  
**Optimization Method:** GridSearchCV with F1 Scoring  
**Best Model by Accuracy:** XGBoost (0.9029)  
**Best Model by Precision:** XGBoost (0.5424)  
**Best Model by Recall:** Logistic Regression (0.7667)
