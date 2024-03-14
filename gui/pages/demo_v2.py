import streamlit as st
import pickle
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
import nltk
from nltk.corpus import stopwords

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
sentence = st.text_input("Enter a sentence:")

# Load the model and TfidfVectorizer from files
with open('extensive_training/RF_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# Using forward slashes
with open('extensive_training/tfid_vectorizer.pkl', 'rb') as file:
    tfid = pickle.load(file)



if st.button("Predict"):
    preprocessed_sentence = transform_text(sentence)
    numerical_features = tfid.transform([preprocessed_sentence]).toarray()
    prediction = clf.predict(numerical_features)
    st.write(f"Model: Random Forest Classification, Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")

