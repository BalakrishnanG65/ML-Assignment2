import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

st.set_page_config(page_title="Bank Marketing Classifier", layout="wide")
st.title("🏦 Bank Marketing Classification App")

# Sidebar for download and model selection
with st.sidebar:
    st.header("Setup")
    
    if st.button("📥 Download Test Data", use_container_width=True):
        test_data = pd.read_csv("data/test_data.csv")
        csv = test_data.to_csv(index=False)
        st.download_button(
            label="📥 Download test_data.csv",
            data=csv,
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True
        )

# Main content
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
    
    # Validate that we have the 'y' column
    if 'y' not in test_df.columns:
        st.error("❌ Error: The uploaded file must contain a 'y' column (target/label)")
        st.stop()
    
    # Test data is already preprocessed and scaled from train_models.py
    # No additional preprocessing needed
    X_test = test_df.drop("y", axis=1).values
    y_test = test_df["y"].values
    
    # Validate feature count
    if X_test.shape[1] != 20:
        st.error(f"""
        ❌ Error: Expected 20 features, but got {X_test.shape[1]}
        
        **Solution:** Please regenerate the test data by:
        1. Delete data/test_data.csv
        2. Run: python train_models.py
        3. Then use the 'Download Test Data' button to get the fresh data
        """)
        st.stop()

    with open(model_files[model_name], "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    st.subheader("📊 Evaluation Metrics")
    
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

    st.subheader("📋 Classification Report")
    
    # Parse classification report into a DataFrame
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.round(4)
    
    st.dataframe(
        report_df.style.background_gradient(cmap='RdYlGn', axis=0).format("{:.4f}"),
        use_container_width=True
    )

    st.subheader("🔲 Confusion Matrix")
    
    # Create a beautiful confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar_kws={'label': 'Count'},
        xticklabels=['Predicted No', 'Predicted Yes'],
        yticklabels=['Actual No', 'Actual Yes'],
        annot_kws={'size': 14, 'weight': 'bold'},
        linewidths=2,
        linecolor='black',
        ax=ax,
        vmin=0
    )
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add detailed confusion matrix interpretation
    with st.expander("ℹ️ Confusion Matrix Details"):
        tn, fp, fn, tp = cm.ravel()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("True Negatives (TN)", tn)
        with col2:
            st.metric("False Positives (FP)", fp)
        with col3:
            st.metric("False Negatives (FN)", fn)
        with col4:
            st.metric("True Positives (TP)", tp)
