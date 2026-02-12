import streamlit as st
import pandas as pd
import pickle
from preprocessing import preprocess_data
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)

st.title("Bank Marketing Classification App")

uploaded_file = st.file_uploader("Upload Test CSV", type="csv")

model_name = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "Random Forest", "KNN", "Naive Bayes", "XGBoost"]
)

model_files = {
    "Logistic Regression": "saved_models/logistic_regression.pkl",
    "Decision Tree": "saved_models/decision_tree.pkl",
    "KNN": "saved_models/knn.pkl",
    "Naive Bayes": "saved_models/naive_bayes.pkl",
    "Random Forest": "saved_models/random_forest.pkl",
    "XGBoost": "saved_models/xgboost.pkl"
}

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    X_test, y_test = preprocess_data(test_df)

    with open(model_files[model_name], "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    st.subheader("Evaluation Metrics")
    
    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Display metrics in columns for better readability
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}")
        st.metric("Precision", f"{precision:.4f}")
    
    with col2:
        st.metric("AUC Score", f"{auc_score:.4f}")
        st.metric("Recall", f"{recall:.4f}")
    
    with col3:
        st.metric("F1 Score", f"{f1:.4f}")
        st.metric("MCC Score", f"{mcc:.4f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))
