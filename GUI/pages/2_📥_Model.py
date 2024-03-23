import streamlit as st
import pickle
import numpy as np
from nltk.stem.porter import PorterStemmer
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import torch

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    processed_text = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english') and word not in string.punctuation]
    return " ".join(processed_text)

# Load the TfidfVectorizer and BERT model
with open('extensive_training/tfid_vectorizer.pkl', 'rb') as file:
    tfid = pickle.load(file)

bert_tokenizer = BertTokenizer.from_pretrained('bert_model_files/')
bert_model = BertForSequenceClassification.from_pretrained('bert_model_files/')

def bert_predict(sentence):
    inputs = bert_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return 'Likely a Spam' if prediction == 1 else 'Likely Not a Spam'

model_paths = {
    'LogisticRegression': 'extensive_training/LR_model.pkl',
    'SupportVectorMachine': 'extensive_training/SVC_model.pkl',
    'MultinomialNB': 'extensive_training/NB_model.pkl',
    'DecisionTreeClassifier': 'extensive_training/DT_model.pkl',
    'AdaBoostClassifier': 'extensive_training/Adaboost_model.pkl',
    'BaggingClassifier': 'extensive_training/Bgc_model.pkl',
    'ExtraTreesClassifier': 'extensive_training/ETC_model.pkl',
    'GradientBoostingClassifier': 'extensive_training/GBDT_model.pkl',
    'XGBClassifier': 'extensive_training/xgb_model.pkl',
    'RandomForestClassifier': 'extensive_training/RF_model.pkl',
    'BERT': 'bert'
}

def predict_spam(sentence):
    preprocessed_sentence = transform_text(sentence)
    numerical_features = tfid.transform([preprocessed_sentence]).toarray()
    results = []
    for model_name, model_path in model_paths.items():
        if model_name == 'BERT':
            prediction_text = bert_predict(sentence)
        else:
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
                prediction = model.predict(numerical_features)
                prediction_text = 'Likely a Spam' if prediction == 1 else 'Likely Not a Spam'
        results.append({'Model': model_name, 'Prediction': prediction_text})
    return results

# Streamlit app
st.title("Spam Detection Demo")
st.markdown("Enter a sentence or upload a CSV file with sentences for spam detection.")

sentence = st.text_input("Enter a sentence:")
uploaded_file = st.file_uploader("Or upload a CSV file with sentences:", type=["csv"])

if st.button("Predict"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Sentence' in df.columns:
            sentences = df['Sentence'].tolist()
        else:
            st.error("CSV file must have a column named 'Sentence'")
            sentences = []
    else:
        sentences = [sentence]

    all_results = []
    for sent in sentences:
        results = predict_spam(sent)
        for result in results:
            result['Sentence'] = sent
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    def highlight_spam(val):
        color = 'darkred' if val == 'Likely a Spam' else 'darkgreen'
        return f'background-color: {color}'

    if len(sentences) == 1 and uploaded_file is None:
        st.dataframe(results_df.drop(columns=['Sentence']).style.applymap(highlight_spam, subset=['Prediction']))
    else:
        st.dataframe(results_df.style.applymap(highlight_spam, subset=['Prediction']))
