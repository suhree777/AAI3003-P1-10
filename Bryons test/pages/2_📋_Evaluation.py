from datasets import load_dataset
import streamlit as st
import pandas as pd

# Load the evaluation metrics
metrics_df = pd.read_csv('extensive_training/classification_metrics.csv')
# Load the dataset
dataset_df = pd.read_csv('extensive_training/cleaned_dataset.csv')
# Display the head of the dataset
st.header('Dataset')
st.dataframe(dataset_df.head(15), width=800, height=300)

# Display the evaluation metrics
st.header('Model Evaluation Metrics')
# Create a table with the model names and evaluation results
st.table(metrics_df.style.format({
    'Model': '{:.2f}',  # Format the model column to two decimal places
    'Accuracy': '{:.2%}',  # Format the accuracy column as a percentage
    'Precision': '{:.2%}',  # Format the precision column as a percentage
    'Recall': '{:.2%}',  # Format the recall column as a percentage
    'F1 Score': '{:.2%}'  # Format the F1 score column as a percentage
}))

