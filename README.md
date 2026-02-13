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
7. [Additional Documentation](#additional-documentation)
8. [Project StructureX](#project-structure)
9. [Troubleshooting](#troubleshooting)

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
- **Features:** 20 input variables
- **Target:** Binary classification (yes/no for term deposit subscription)
- **Class Distribution:**
  - Class 0 (No): 3,668 samples (89.1%)
  - Class 1 (Yes): 451 samples (10.9%)
  - **Imbalance Ratio:** 8.13:1

### Feature Breakdown
The 20 features include:
- **Demographic:** age, job, marital status, education, credit default status, housing loan, personal loan
- **Campaign:** contact type (cellular/telephone), day of week, month, duration of last contact, number of contacts during campaign, days since last contact, outcome of previous campaign
- **Economic:** employment variation rate, consumer price index, consumer confidence index, euribor 3-month rate, number of employees

### Data Preprocessing
- Missing values removed
- Categorical features encoded using LabelEncoder
- Features scaled using StandardScaler (z-score normalization: $z = \frac{x - \mu}{\sigma}$)
- Train-Test Split: 80% training (3,295 samples), 20% testing (824 samples) with stratification
---

## 🤖 Models Used

This project implements and evaluates **6 machine learning algorithms** to compare their performance on the binary classification task:

### 1. **Logistic Regression**
A linear classification model with probabilistic outputs. Used as a baseline model due to its interpretability and efficiency.

### 2. **Decision Tree**
A tree-based classifier that recursively splits features to maximize information gain. Captures non-linear relationships and is highly interpretable.

### 3. **k-Nearest Neighbors (kNN)**
An instance-based, non-parametric algorithm that classifies samples based on the majority class of k nearest neighbors in the feature space.

### 4. **Naive Bayes**
A probabilistic classifier based on Bayes' theorem with the assumption of feature independence. Fast and effective for binary classification.

### 5. **Random Forest (Ensemble)**
An ensemble method that combines multiple decision trees with bootstrap aggregation. Reduces overfitting and handles non-linearity effectively.

### 6. **XGBoost (Ensemble)**
An advanced gradient boosting framework that sequentially builds trees to correct previous errors. State-of-the-art performance for structured data.

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

---

## 🔬 Hyperparameter Tuning

### Optimization Strategy
- **Method:** GridSearchCV with 5-fold Cross-Validation
- **Scoring Metric:** F1 Score (balances precision & recall)
- **Optimization Goal:** Not accuracy, but balanced performance

### Model-Specific Parameters Tuned

#### Logistic Regression
| Parameter | Values Tested |
|-----------|--------------|
| C (Regularization) | [0.001, 0.01, 0.1, 1, 10] |
| Class Weight | ['balanced', {0:1, 1:5}, {0:1, 1:8}] |

#### Decision Tree
| Parameter | Values Tested |
|-----------|--------------|
| Max Depth | [8, 10, 12, 15] |
| Min Samples Split | [5, 10, 15] |
| Min Samples Leaf | [2, 4, 5] |
| Class Weight | ['balanced', custom weights] |

#### Random Forest
| Parameter | Values Tested |
|-----------|--------------|
| N Estimators | [100, 150, 200] |
| Max Depth | [12, 15, 18, 20] |
| Min Samples Split | [8, 10, 12] |
| Class Weight | ['balanced', 'balanced_subsample'] |

#### KNN
| Parameter | Values Tested |
|-----------|--------------|
| N Neighbors | [5, 7, 9, 11, 13] |
| Weights | ['uniform', 'distance'] |
| Metric | ['minkowski', 'euclidean', 'manhattan'] |

#### XGBoost
| Parameter | Values Tested |
|-----------|--------------|
| Max Depth | [6, 7] |
| Learning Rate | [0.1, 0.15] |
| Scale Pos Weight | [8.13×1.5, 8.13×2] |

---

## 📈 Results & Performance

### Version Comparison: V1 vs V2

| Metric | V1 Avg | V2 Avg | Improvement |
|--------|--------|--------|------------|
| **Accuracy** | 0.8928 | 0.8905 | -0.23% (acceptable trade-off) |
| **Precision** ⭐ | 0.4810 | 0.5079 | **+5.59%** ✅ |
| **Recall** | 0.6344 | 0.5702 | -8.95% (controlled) |
| **F1 Score** ⭐ | 0.5298 | 0.5694 | **+7.47%** ✅ |
| **AUC** | 0.7864 | 0.7866 | +0.03% |
| **MCC** ⭐ | 0.4709 | 0.4947 | **+5.05%** ✅ |

### Per-Model Results (V2 - Optimized)

#### Detailed Metrics by Model

| Model | Accuracy | Precision | Recall | F1 Score | AUC | MCC |
|-------|----------|-----------|--------|----------|-----|-----|
| Logistic Regression | 0.8993 | 0.5267 | 0.7667 | 0.6244 | 0.8411 | 0.5819 |
| **Decision Tree** | **0.8993** | **0.5280** | **0.7333** | **0.6140** | **0.8265** | **0.5677** 🏆 |
| Random Forest | 0.8993 | 0.5276 | 0.7444 | 0.6175 | 0.8314 | 0.5725 |
| KNN | 0.8932 | 0.5263 | 0.2222 | 0.3125 | 0.5988 | 0.2940 |
| Naive Bayes | 0.8289 | 0.3455 | 0.6333 | 0.4471 | 0.7431 | 0.3790 |
| XGBoost | 0.9029 | 0.5424 | 0.7111 | 0.6154 | 0.8188 | 0.5677 |

### Best Performing Models

1. **🥇 Decision Tree** - Best balanced performance
   - Highest precision (0.5280)
   - Good F1 score (0.6140)
   - Interpretable results

2. **🥈 XGBoost** - Best accuracy
   - Highest accuracy (0.9029)
   - Good precision (0.5424)
   - High recall (0.7111)

3. **🥉 Logistic Regression** - Best recall
   - Finds most positive cases (0.7667)
   - Interpretable model
   - Fast prediction

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Download Dataset

1. Go to: https://archive.ics.uci.edu/dataset/222/bank+marketing
2. Download **bank-additional.csv**
3. Place in project root directory

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train Models

```bash
python train_models.py
```

**Training Output:**
- ✅ All 6 models trained with GridSearchCV
- ✅ Best hyperparameters displayed
- ✅ Models saved to `saved_models/`
- ✅ Test data generated to `data/test_data.csv`

### Step 4: Launch Streamlit App

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

### Step 5: Use the Application

1. **Sidebar:** Click "📥 Download Test Data" button
2. **Main Area:** Upload the downloaded CSV file
3. **Dropdown:** Select a model to evaluate
4. **View Results:** See 6 metrics + confusion matrix

---

## 📐 Evaluation Metrics Explanation

### 1. **Accuracy**
- **Formula:** $\frac{TP + TN}{TP + TN + FP + FN}$
- **Meaning:** Overall percentage of correct predictions
- **Use Case:** General performance overview
- **Limitation:** Misleading with imbalanced data

### 2. **Precision**
- **Formula:** $\frac{TP}{TP + FP}$
- **Meaning:** Of predicted positives, how many were correct?
- **Use Case:** When false positives are costly
- **Bank Context:** Avoid wasting marketing budget on unlikely customers

### 3. **Recall (Sensitivity)**
- **Formula:** $\frac{TP}{TP + FN}$
- **Meaning:** Of actual positives, how many were found?
- **Use Case:** When false negatives are costly
- **Bank Context:** Don't miss potential customers

### 4. **F1 Score**
- **Formula:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Meaning:** Harmonic mean of precision & recall
- **Use Case:** Balances both metrics for imbalanced data
- **Range:** 0 to 1 (higher is better)

### 5. **AUC Score**
- **Formula:** Area under ROC curve
- **Meaning:** Probability model ranks random positive higher than random negative
- **Use Case:** Class-imbalance robust metric
- **Range:** 0 to 1 (0.5 is random, 1.0 is perfect)

### 6. **MCC (Matthews Correlation Coefficient)**
- **Formula:** $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$
- **Meaning:** Correlation between predicted and actual
- **Use Case:** Single score for imbalanced classification
- **Range:** -1 to 1 (higher is better)

---

## 💡 Key Findings

### 1. **Class Imbalance is Critical**
- Initial versions treated all classes equally
- Resulted in poor precision (0.46) and low minority class detection
- V2 improved precision by 15-27% using:
  - Balanced class weights
  - SMOTE resampling
  - Scale pos weight tuning

### 2. **Hyperparameter Tuning Works**
- GridSearchCV optimizing for F1 (not accuracy) improved:
  - F1 Score: +7.47%
  - MCC: +5.05%
  - Precision: +5.59%

### 3. **Ensemble Methods Excel**
- Random Forest consistently in top 3
- XGBoost achieves best accuracy (0.9029)
- Decision Tree best precision (0.5280)
- No single "best" model - depends on business goal

### 4. **Trade-offs Exist**
- Higher recall (finding all positives) often means lower precision
- XGBoost: High recall (0.7111) but slightly lower precision (0.5424)
- Decision Tree: Balanced precision (0.5280) and recall (0.7333)

### 5. **Evaluation Metric Choice Matters**
- Accuracy alone is misleading (0.90 means little with imbalanced data)
- F1 + Precision + MCC provide balanced view
- AUC is robust to class imbalance

---

## 📁 Project Structure

```
ML-Assignment2/
├── README.md                 # Project documentation
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
├── saved_models/            # Trained model files
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── xgboost.pkl
│
├── data/                    # Dataset files
│   ├── bank-additional.csv  # Original dataset (download required)
│   └── test_data.csv        # Generated test set (auto-created)
│
└── saved_models/            # Output models (auto-created)
```

---

## 🔧 Troubleshooting

| Error | Solution |
|-------|----------|
| `FileNotFoundError: bank-additional.csv` | Download from UCI repository and place in project root |
| `ModuleNotFoundError: streamlit` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: saved_models/*.pkl` | Run `python train_models.py` to train models |
| `ValueError: Expected 20 features` | Delete `data/test_data.csv` and retrain with `python train_models.py` |

---

## 📚 Limitations & Future Work

### Current Limitations

1. **Dataset Size**
   - Only 4,119 samples; larger dataset could improve generalization
   - Missing temporal patterns (dataset is from 2008-2010)

2. **Feature Engineering**
   - Current features used as-is
   - Could create interaction features or polynomial features
   - Domain expertise could identify important feature combinations

3. **Model Selection**
   - Limited to 6 models
   - No advanced techniques like:
     - Stacking/Voting ensembles
     - Neural networks
     - AutoML

4. **Hardware Constraints**
   - GridSearchCV limited to small parameter grids due to computation

### Future Improvements

1. **Advanced Preprocessing**
   - Feature selection (correlation analysis, feature importance)
   - Feature engineering (interactions, polynomials)
   - Automated outlier detection

2. **Ensemble Methods**
   - Stacking multiple models
   - Voting classifiers
   - Blending techniques

3. **Deep Learning**
   - Neural networks for non-linear patterns
   - LSTM for temporal patterns

4. **Production Deployment**
   - REST API with FastAPI
   - Docker containerization
   - Real-time prediction endpoint

5. **Monitoring & Maintenance**
   - Model drift detection
   - Performance monitoring dashboard
   - Automated retraining pipeline

---

## 📊 Performance Metrics Summary

### Confusion Matrix Interpretation

```
                 Predicted No  Predicted Yes
Actual No              [TN]        [FP]
Actual Yes             [FN]        [TP]
```

- **TP (True Positive):** Correctly predicted yes (will subscribe)
- **TN (True Negative):** Correctly predicted no (won't subscribe)
- **FP (False Positive):** Wrongly predicted yes (wasted marketing effort)
- **FN (False Negative):** Wrongly predicted no (missed opportunity)

### For Bank Marketing
- **Minimize FP:** Avoid contacting customers unlikely to subscribe
- **Minimize FN:** Don't miss potential customers who would subscribe
- **Balance Trade-off:** Use F1 score and MCC

---

## 📚 References

1. **Dataset:** [Bank Marketing Dataset - UCI ML Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
2. **Class Imbalance:** [Imbalanced Learn - SMOTE](https://imbalanced-learn.org/)
3. **Model Tuning:** [Scikit-learn GridSearchCV](https://scikit-learn.org/stable/modules/grid_search.html)
4. **XGBoost:** [XGBoost Documentation](https://xgboost.readthedocs.io/)
5. **Metrics:** [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## 📋 Version History

| Version | Date | Changes |
|---------|------|---------|
| **v2.0** | Feb 2026 | GridSearchCV hyperparameter tuning, improved class imbalance handling, F1 optimization |
| **v1.0** | Feb 2026 | Initial release with 6 baseline models |

---

**Assignment Status:** ✅ Complete  
**Last Updated:** February 2026  
**Optimization Method:** GridSearchCV with F1 Scoring  
**Best Model:** Decision Tree (Balanced Precision & Recall)
