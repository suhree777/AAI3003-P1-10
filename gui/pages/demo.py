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

# Load the TfidfVectorizer from files
with open('extensive_training/tfid_vectorizer.pkl', 'rb') as file:
    tfid = pickle.load(file)

# Define a dictionary mapping model names to their file paths
model_paths = {
    'LogisticRegression': 'extensive_training/LR_model.pkl',
    'SupportVectorMachine': 'extensive_training/SVC_model.pkl',
    'MultinomialNB': 'extensive_training/NB_model.pkl',
    'DecisionTreeClassifiern': 'extensive_training/DT_model.pkl',
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

    # Iterate over the models and perform predictions
    for model_name, model_path in model_paths.items():
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
            prediction = model.predict(numerical_features)
            st.write(f"Model: {model_name}, Prediction: {'Spam' if prediction == 1 else 'Not Spam'}")
