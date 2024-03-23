import streamlit as st
import pandas as pd
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Load stopwords once
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    processed_text = []
    for word in text:
        if word.isalnum() and word not in stop_words and word not in string.punctuation:
            processed_text.append(ps.stem(word))
    return " ".join(processed_text)

# Load TfidfVectorizer
with open('extensive_training/tfid_vectorizer.pkl', 'rb') as file:
    tfid = pickle.load(file)

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert_model_files/')
bert_model = BertForSequenceClassification.from_pretrained('bert_model_files/')

def bert_predict(sentence):
    inputs = bert_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return 'Likely a Spam' if prediction == 1 else 'Likely Not a Spam'

# Streamlit app
st.title("Spam Detection Demo")
st.markdown(
    """
    Here we can demonstrate using the models to predict if the input sentence is a spam or not.
    Please input a sentence or upload a CSV file containing sentences, and it will run through the models and provide a prediction.
    """
)
sentence = st.text_input("Enter a sentence:")
uploaded_file = st.file_uploader("OR Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    if 'Sentence' in data.columns:
        st.write(data)
    else:
        st.error("Uploaded CSV file must contain a 'Sentence' column.")

if st.button("Predict"):
    if uploaded_file is not None and 'Sentence' in data.columns:
        data['Prediction'] = data['Sentence'].apply(lambda x: bert_predict(transform_text(x)))
        st.write(data)
    else:
        preprocessed_sentence = transform_text(sentence)
        numerical_features = tfid.transform([preprocessed_sentence]).toarray()
        results_df = pd.DataFrame(columns=['Model', 'Prediction'])

        for model_name, model_path in model_paths.items():
            if model_name == 'BERT':
                prediction_text = bert_predict(sentence)
            else:
                with open(model_path, 'rb') as model_file:
                    model = pickle.load(model_file)
                    prediction = model.predict(numerical_features)
                    prediction_text = 'Likely a Spam' if prediction == 1 else 'Likely Not a Spam'
            new_row_df = pd.DataFrame({'Model': [model_name], 'Prediction': [prediction_text]})
            results_df = pd.concat([results_df, new_row_df], ignore_index=True)

        def highlight_spam(val):
            color = 'darkred' if val == 'Likely a Spam' else 'darkgreen'
            return 'background-color: %s' % color

