import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from devs.dataset import save_dataset  # Import the get_dataset function


def evaluate_model(model, dataset):
    X_test = dataset['test']['text']
    y_test = dataset['test']['label']

    predictions = model.predict(X_test)

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

def main():
    st.header("Model Evaluation")

    # Check if a model has been uploaded
    if 'uploaded_model' in st.session_state:
        # Load the predefined dataset
        dataset = save_dataset()

        # Load the uploaded model
        model = pickle.load(st.session_state['uploaded_model'])

        # Evaluate the model
        performance_metrics = evaluate_model(model, dataset)

        # Convert the metrics dictionary to a DataFrame for display
        metrics_df = pd.DataFrame(performance_metrics)

        # Display the performance evaluation analysis in a table
        st.write("### Performance Evaluation Analysis")
        st.table(metrics_df)
    else:
        # Display a message asking the user to upload a model in the 2_📥_Model.py page
        st.write("#### Please upload at least one model in", "<span style='color: red;'>📥 Model</span>.", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
