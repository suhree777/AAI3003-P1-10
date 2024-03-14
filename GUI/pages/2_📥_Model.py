import streamlit as st
import pickle
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    processed_text = []
    for word in text:
        if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation:
            processed_text.append(ps.stem(word))
    return " ".join(processed_text)

# Streamlit app
st.title("Spam Detection Demo")
st.markdown(
    """
    Here we can demonstrate using the models to predict if the input sentence is a spam or not.
    Please input a sentence and it will run through the models and provide a prediction.
    """
)
sentence = st.text_input("Enter a sentence:")

# Load the TfidfVectorizer from files
with open('extensive_training/tfid_vectorizer.pkl', 'rb') as file:
    tfid = pickle.load(file)

# Define a dictionary mapping model names to their file paths
model_paths = {
    'LogisticRegression': 'extensive_training/LR_model.pkl',
    'SupportVectorMachine': 'extensive_training/SVC_model.pkl',
    'MultinomialNB': 'extensive_training/NB_model.pkl',
    'DecisionTreeClassifier': 'extensive_training/DT_model.pkl',  # Corrected here
    'AdaBoostClassifier': 'extensive_training/Adaboost_model.pkl',
    'BaggingClassifier': 'extensive_training/Bgc_model.pkl',
    'ExtraTreesClassifier': 'extensive_training/ETC_model.pkl',
    'GradientBoostingClassifier': 'extensive_training/GBDT_model.pkl',
    'XGBClassifier': 'extensive_training/xgb_model.pkl',
    'RandomForestClassifier': 'extensive_training/RF_model.pkl'
}

if st.button("Predict"):
    preprocessed_sentence = transform_text(sentence)
    numerical_features = tfid.transform([preprocessed_sentence]).toarray()

    # Create a DataFrame to store the results
    results_df = pd.DataFrame(columns=['Model', 'Prediction'])

    # Iterate over the models and perform predictions
    for model_name, model_path in model_paths.items():
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            prediction = model.predict(numerical_features)
            prediction_text = 'Likely a Spam' if prediction == 1 else 'Likely Not a Spam'
            # Create a new DataFrame for the current prediction and concatenate it
            new_row_df = pd.DataFrame({'Model': [model_name], 'Prediction': [prediction_text]})
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)

    # Define a function to apply color based on the prediction value
    def highlight_spam(val):
        color = 'darkred' if val == 'Likely a Spam' else 'darkgreen'
        return 'background-color: %s' % color

    # Apply the highlighting function only to the 'Prediction' column
    st.dataframe(results_df.style.applymap(highlight_spam, subset=pd.IndexSlice[:, ['Prediction']]), width=500)