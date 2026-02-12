import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # Parse classification report into a DataFrame
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.round(4)
    
    # Create a more readable format
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Detailed Classification Report:**")
        # Display as a dataframe with styling
        st.dataframe(
            report_df.style.background_gradient(cmap='RdYlGn', axis=0).format("{:.4f}"),
            use_container_width=True
        )
    
    with col2:
        st.write("**Summary:**")
        summary_data = {
            "Metric": ["Precision", "Recall", "F1-Score"],
            "Class 0": [f"{report_dict['0']['precision']:.4f}", f"{report_dict['0']['recall']:.4f}", f"{report_dict['0']['f1-score']:.4f}"],
            "Class 1": [f"{report_dict['1']['precision']:.4f}", f"{report_dict['1']['recall']:.4f}", f"{report_dict['1']['f1-score']:.4f}"]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

    st.subheader("Confusion Matrix")
    
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
    with st.expander("📊 Confusion Matrix Interpretation"):
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
        
        st.markdown("---")
        st.markdown(f"""
        **Interpretation:**
        - **TN ({tn})**: Correctly predicted No/Negative cases
        - **FP ({fp})**: Incorrectly predicted Yes/Positive (Type I Error)
        - **FN ({fn})**: Incorrectly predicted No/Negative (Type II Error)
        - **TP ({tp})**: Correctly predicted Yes/Positive cases
        """)
