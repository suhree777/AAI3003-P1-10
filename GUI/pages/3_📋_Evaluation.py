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

# Identify the best model based on the highest average of metrics
best_model = metrics_df.set_index('Classifier').mean(axis=1).idxmax()

def plot_accuracy():
    st.write("### Accuracy of Models")
    accuracy_chart_data = metrics_df[['Classifier', 'Accuracy']].set_index('Classifier')
    st.bar_chart(accuracy_chart_data)

def plot_precision():
    st.write("### Precision of Models")
    precision_chart_data = metrics_df[['Classifier', 'Precision']].set_index('Classifier')
    st.bar_chart(precision_chart_data)

def plot_recall():
    st.write("### Recall of Models")
    recall_chart_data = metrics_df[['Classifier', 'Recall']].set_index('Classifier')
    st.bar_chart(recall_chart_data)

def plot_f1_score():
    st.write("### F1-score of Models")
    f1_chart_data = metrics_df[['Classifier', 'F1']].set_index('Classifier')
    st.bar_chart(f1_chart_data)

def display_best_model():
    classifier_full_names = {
        "SVC": "Support Vector Classifier",
        "KNN": "K-Nearest Neighbors",
        "NB": "Naive Bayes",
        "DT": "Decision Tree",
        "LR": "Logistic Regression",
        "RF": "Random Forest",
        "Adaboost": "Adaptive Boosting",
        "Bgc": "Bagging Classifier",
        "ETC": "Extra Trees Classifier",
        "GBDT": "Gradient Boosting Decision Tree",
        "xgb": "XGBoost"
    }

    best_model_full_name = classifier_full_names.get(best_model, best_model)
    st.info(f"Based on the average of the metrics, the best model is: **{best_model}** ({best_model_full_name})")


def main():
    display_best_model()
    plot_accuracy()
    plot_precision()
    plot_recall()
    plot_f1_score()

if __name__ == "__main__":
    main()