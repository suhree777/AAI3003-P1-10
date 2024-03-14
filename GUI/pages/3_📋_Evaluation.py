from datasets import load_dataset
import streamlit as st
import pandas as pd

# Load the evaluation metrics
metrics_df = pd.read_csv('extensive_training/classification_metrics.csv')
# Load the dataset
dataset_df = pd.read_csv('extensive_training/cleaned_dataset.csv')
# Display the head of the dataset
st.title("Model Evaluations")
st.markdown(
    """
    For our model evaluation and training, preprocessing is done on the dataset before training the models. 
    The preprocessing steps include:

    - **Removing Null Values:** Ensuring that there are no missing values in the dataset.
    - **Removing Duplicates:** Eliminating duplicate entries to prevent bias in the model.
    - **Lowercasing:** Converting all text to lowercase to ensure uniformity.
    - **Tokenization:** Splitting the text into individual words or tokens.
    - **Removing Special Characters:** Eliminating characters that are not alphanumeric to simplify the text.
    - **Removing Stopwords and Punctuation:** Discarding common words and punctuation marks that do not contribute to the meaning of the text.
    - **Stemming:** Reducing words to their base or root form to consolidate similar variations of a word.
    """
)

st.header('Dataset')
st.dataframe(dataset_df.head(15), width=1100, height=300)
st.markdown(
    """
    In our spam detection model, a TF-IDF Vectorizer is used to convert the preprocessed text data into numerical features. This vectorization technique calculates the term frequency and inverse document frequency for each word in the dataset to represent the text data as a matrix of numerical values. This allows us to capture the importance of each word in relation to the document and the entire corpus. The TF-IDF Vectorizer not only helps in reducing the dimensionality of the text data but also enhances the model's ability to distinguish between spam and non-spam messages based on the significance of the words used.
    """
)
# Display the evaluation metrics
st.header('Model Evaluation Metrics')
st.markdown(
    """
    After training, we used sklearn accuracy_score, precision_score, recall_score and f1_score to view the evaluation scores of each model.
    """
)
# Create a table with the model names and evaluation results
st.table(metrics_df.style.format({
    'Model': '{:.2f}',  # Format the model column to two decimal places
    'Accuracy': '{:.2%}',  # Format the accuracy column as a percentage
    'Precision': '{:.2%}',  # Format the precision column as a percentage
    'Recall': '{:.2%}',  # Format the recall column as a percentage
    'F1 Score': '{:.2%}'  # Format the F1 score column as a percentage
}))

