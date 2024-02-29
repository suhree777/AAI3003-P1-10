import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set page configuration
st.set_page_config(page_title="AAI3003", layout="wide", page_icon="📥")

# Load your predefined dataset
# Make sure to adjust the path to your dataset and the column names
dataset = pd.read_csv('path/to/your/predefined_dataset.csv')
X_test = dataset['text_column']  # Adjust 'text_column' to your dataset's text column name
y_test = dataset['label_column']  # Adjust 'label_column' to your dataset's label column name

# Allow users to upload a model
uploaded_model = st.file_uploader("Upload your spam detection model", type=["pkl"])

if uploaded_model is not None:
    # Load the uploaded model
    model = pickle.load(uploaded_model)

    # Run the model on the predefined dataset
    predictions = model.predict(X_test)

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label='spam')  # Adjust pos_label if necessary
    recall = recall_score(y_test, predictions, pos_label='spam')  # Adjust pos_label if necessary
    f1 = f1_score(y_test, predictions, pos_label='spam')  # Adjust pos_label if necessary

    # Store the metrics in a dictionary
    performance_metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    # Save the performance metrics to a session state for access in 3_📋_Evaluation.py
    st.session_state['model_performance'] = performance_metrics

    # Display the performance metrics
    st.write("Model Performance Metrics:")
    st.write(performance_metrics)
else:
    st.write("Please upload a model to evaluate its performance.")
