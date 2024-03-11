import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from devs.dataset import save_dataset  # Import the save_dataset function
from transformers import pipeline
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Instantiate the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_predictions(model, X_test, tokenizer=None):
    if tokenizer is not None:
        inputs = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=1).tolist()
    else:
        predictions = model(X_test)
    return predictions

def evaluate_model(model, dataset, tokenizer=None):
    X = dataset['train']['sms']
    y = dataset['train']['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    predictions = get_predictions(model, X_test, tokenizer)
    
    # Check for NaN values in y_test and predictions
    nan_indices_y_test = np.isnan(y_test)
    nan_indices_predictions = np.isnan(predictions)

    # Replace NaN values with a valid label (e.g., 0 or 1)
    y_test[nan_indices_y_test] = 0  # Replace NaN with 0
    predictions[nan_indices_predictions] = 0  # Replace NaN with 0

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label=1)
    recall = recall_score(y_test, predictions, pos_label=1)
    f1 = f1_score(y_test, predictions, pos_label=1)

    performance_metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    }

    return performance_metrics

def load_model(model_key):
    model_path = st.session_state[model_key].name

    if 'safetensors' in model_path:
        # Load the model using the Transformers library
        model = pipeline("text-classification", model=model_path)
    else:
        # Load the model using torch.load() for PyTorch models
        model = torch.load(model_path)

    return model

def main():
    st.header("Model Evaluation")

    # Check if any models have been uploaded
    if 'models' in st.session_state:
        # Load the dataset from devs/dataset.py
        dataset = save_dataset()
        # Iterate over each uploaded model and evaluate it
        for model_name, model in st.session_state['models'].items():
            try:
                # Evaluate the model
                performance_metrics = evaluate_model(model, dataset)

                # Convert the metrics dictionary to a DataFrame for display
                metrics_df = pd.DataFrame(performance_metrics)

                # Display the model name and the performance evaluation analysis in a table
                st.write(f"### Model: {model_name}")
                st.table(metrics_df)
            except ValueError as e:
                st.write(str(e))
    else:
        # Display a message asking the user to upload a model in the 2_📥_Model.py page
        st.write("#### Please upload at least one model in", "<span style='color: red;'>📥 Model</span>.", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
