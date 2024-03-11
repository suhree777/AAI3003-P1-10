import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from devs.dataset import save_dataset  # Import the save_dataset function
from transformers import pipeline
import torch
from sklearn.model_selection import train_test_split

def evaluate_model(model, dataset):
    X = dataset['train']['text']
    y = dataset['train']['label']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if isinstance(model, pipeline):
        predictions = [result['label'] for result in model(X_test)]
    elif isinstance(model, torch.nn.Module):
        # Handle evaluation for PyTorch models
        predictions = []  # Replace this with actual PyTorch model evaluation
    else:
        # Handle other types of models
        predictions = []  # Replace this with code to use the model for prediction

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label=1)
    recall = recall_score(y_test, predictions, pos_label=1)
    f1 = f1_score(y_test, predictions, pos_label=1)

    # Store the metrics in a dictionary
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
