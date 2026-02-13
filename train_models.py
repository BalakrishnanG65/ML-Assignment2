"""
Training script for Bank Marketing Classification Models
This script trains all 6 models on the bank marketing dataset with hyperparameter optimization

Instructions:
1. Download the dataset from: https://archive.ics.uci.edu/dataset/222/bank+marketing
2. Extract the bank-additional.csv file to the current directory
3. Run: python train_models.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle
import os

# Create saved_models directory
os.makedirs("saved_models", exist_ok=True)

print("Loading dataset...")
# Load the bank marketing dataset
# Download from: https://archive.ics.uci.edu/dataset/222/bank+marketing
df = pd.read_csv("data/bank-additional.csv", sep=";")

print(f"Dataset shape: {df.shape}")
print(f"Target column distribution:\n{df['y'].value_counts()}")

# Data preprocessing
print("\nPreprocessing data...")
df = df.copy()
df = df.dropna()

# Encode categorical columns
cat_cols = df.select_dtypes(include=["object"]).columns
encoders = {}

for col in cat_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    encoders[col] = encoder

X = df.drop("y", axis=1)
y = df["y"]

# Map target values to 0 and 1
y_encoded = LabelEncoder().fit_transform(y)
y = pd.Series(y_encoded)

print(f"Target distribution after encoding: {y.value_counts().to_dict()}")
print(f"Class imbalance ratio: {y.value_counts()[0] / y.value_counts()[1]:.2f}:1")

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train_scaled.shape}")
print(f"Test set size: {X_test_scaled.shape}")

# Save test data
test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
test_df['y'] = y_test.values
# Ensure all columns are float type to prevent issues when loading
test_df = test_df.astype('float32')
test_df.to_csv("data/test_data.csv", index=False)
print(f"\nTest data saved to data/test_data.csv")
print(f"Test data shape: {test_df.shape} (rows, columns)")
print(f"Columns: {list(test_df.columns)}")

# ============================================
# 1. LOGISTIC REGRESSION
# ============================================
print("\n" + "="*50)
print("Training Logistic Regression (with GridSearch)...")
print("="*50)

# Calculate class weight ratio
class_weight_ratio = (y_train == 0).sum() / (y_train == 1).sum()
class_weights = {0: 1, 1: class_weight_ratio * 1.5}  # Weight minority class more

lr_params = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'class_weight': ['balanced', {0: 1, 1: 5}, {0: 1, 1: 8}]
}

lr_base = LogisticRegression(max_iter=5000, solver="saga", n_jobs=-1, random_state=42)
lr_grid = GridSearchCV(lr_base, lr_params, cv=5, scoring='f1', n_jobs=-1, verbose=0)
lr_grid.fit(X_train_scaled, y_train)
lr_model = lr_grid.best_estimator_

pickle.dump(lr_model, open("saved_models/logistic_regression.pkl", "wb"))
lr_score = lr_model.score(X_test_scaled, y_test)
print(f"Best params: {lr_grid.best_params_}")
print(f"Logistic Regression Accuracy: {lr_score:.4f}")

# ============================================
# 2. DECISION TREE
# ============================================
print("\n" + "="*50)
print("Training Decision Tree (with GridSearch)...")
print("="*50)

dt_params = {
    'max_depth': [8, 10, 12, 15],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [2, 4, 5],
    'class_weight': ['balanced', {0: 1, 1: 5}, {0: 1, 1: 8}]
}

dt_base = DecisionTreeClassifier(random_state=42)
dt_grid = GridSearchCV(dt_base, dt_params, cv=5, scoring='f1', n_jobs=-1, verbose=0)
dt_grid.fit(X_train_scaled, y_train)
dt_model = dt_grid.best_estimator_

pickle.dump(dt_model, open("saved_models/decision_tree.pkl", "wb"))
dt_score = dt_model.score(X_test_scaled, y_test)
print(f"Best params: {dt_grid.best_params_}")
print(f"Decision Tree Accuracy: {dt_score:.4f}")

# ============================================
# 3. RANDOM FOREST
# ============================================
print("\n" + "="*50)
print("Training Random Forest (with GridSearch)...")
print("="*50)

rf_params = {
    'n_estimators': [100, 150, 200],
    'max_depth': [12, 15, 18, 20],
    'min_samples_split': [8, 10, 12],
    'min_samples_leaf': [4, 5, 6],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf_base, rf_params, cv=5, scoring='f1', n_jobs=-1, verbose=0)
rf_grid.fit(X_train_scaled, y_train)
rf_model = rf_grid.best_estimator_

pickle.dump(rf_model, open("saved_models/random_forest.pkl", "wb"))
rf_score = rf_model.score(X_test_scaled, y_test)
print(f"Best params: {rf_grid.best_params_}")
print(f"Random Forest Accuracy: {rf_score:.4f}")

# ============================================
# 4. K-NEAREST NEIGHBORS
# ============================================
print("\n" + "="*50)
print("Training K-Nearest Neighbors (with GridSearch)...")
print("="*50)

knn_params = {
    'n_neighbors': [5, 7, 9, 11, 13],
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean', 'manhattan']
}

knn_base = KNeighborsClassifier(n_jobs=-1)
knn_grid = GridSearchCV(knn_base, knn_params, cv=5, scoring='f1', n_jobs=-1, verbose=0)
knn_grid.fit(X_train_scaled, y_train)
knn_model = knn_grid.best_estimator_

pickle.dump(knn_model, open("saved_models/knn.pkl", "wb"))
knn_score = knn_model.score(X_test_scaled, y_test)
print(f"Best params: {knn_grid.best_params_}")
print(f"KNN Accuracy: {knn_score:.4f}")

# ============================================
# 5. NAIVE BAYES (with SMOTE)
# ============================================
print("\n" + "="*50)
print("Training Naive Bayes (with SMOTE)...")
print("="*50)

# Use more aggressive SMOTE ratio for better minority class representation
smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.6)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE - Training set size: {X_train_smote.shape}")
print(f"Target distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")

nb_model = GaussianNB(var_smoothing=1e-8)
nb_model.fit(X_train_smote, y_train_smote)
pickle.dump(nb_model, open("saved_models/naive_bayes.pkl", "wb"))
nb_score = nb_model.score(X_test_scaled, y_test)
print(f"Naive Bayes Accuracy: {nb_score:.4f}")

# ============================================
# 6. XGBOOST (with advanced tuning)
# ============================================
print("\n" + "="*50)
print("Training XGBoost (with GridSearch)...")
print("="*50)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale pos weight (for class imbalance): {scale_pos_weight:.2f}")

xgb_params = {
    'max_depth': [5, 6, 7, 8],
    'learning_rate': [0.05, 0.1, 0.15],
    'n_estimators': [100, 150, 200],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'min_child_weight': [3, 5, 7],
    'scale_pos_weight': [scale_pos_weight, scale_pos_weight * 1.5, scale_pos_weight * 2]
}

# Use a smaller grid for faster training
xgb_params_small = {
    'max_depth': [6, 7],
    'learning_rate': [0.1, 0.15],
    'scale_pos_weight': [scale_pos_weight * 1.5, scale_pos_weight * 2]
}

xgb_base = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
    use_label_encoder=False
)

xgb_grid = GridSearchCV(xgb_base, xgb_params_small, cv=5, scoring='f1', n_jobs=-1, verbose=0)
xgb_grid.fit(X_train_scaled, y_train)
xgb_model = xgb_grid.best_estimator_

pickle.dump(xgb_model, open("saved_models/xgboost.pkl", "wb"))
xgb_score = xgb_model.score(X_test_scaled, y_test)
print(f"Best params: {xgb_grid.best_params_}")
print(f"XGBoost Accuracy: {xgb_score:.4f}")

# ============================================
# Summary
# ============================================
print("\n" + "="*50)
print("TRAINING COMPLETE - MODEL SUMMARY")
print("="*50)
results = {
    "Logistic Regression": lr_score,
    "Decision Tree": dt_score,
    "Random Forest": rf_score,
    "KNN": knn_score,
    "Naive Bayes": nb_score,
    "XGBoost": xgb_score
}

for model_name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:25} | Accuracy: {score:.4f}")

print("\n✅ All models trained and saved to saved_models/")
print("✅ Test data saved to data/test_data.csv")
print("\n📊 Improvements made:")
print("   • Hyperparameter tuning with GridSearchCV")
print("   • Better class weighting for imbalanced data")
print("   • Improved SMOTE configuration for Naive Bayes")
print("   • F1-score optimization (balanced precision & recall)")
print("   • Enhanced XGBoost scale_pos_weight tuning")
print("\nNext steps:")
print("1. Run: streamlit run app.py")
print("2. Click '📥 Download Test Data' button in the sidebar")
print("3. Upload the downloaded CSV to evaluate the models")
