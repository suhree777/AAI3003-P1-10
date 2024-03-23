import streamlit as st
import pickle
import numpy as np
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

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
    Please input a sentence or upload a CSV file with sentences, and it will run through the models and provide predictions.
    """
)
sentence = st.text_input("Enter a sentence:")
uploaded_file = st.file_uploader("Or upload a CSV file with sentences:", type=["csv"])

# Load the TfidfVectorizer from files
with open('extensive_training/tfid_vectorizer.pkl', 'rb') as file:
    tfid = pickle.load(file)

# Load the BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert_model_files/')
bert_model = BertForSequenceClassification.from_pretrained('bert_model_files/')

def bert_predict(sentence):
    # Tokenize and encode the sentence
    inputs = bert_tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Get the prediction from the BERT model
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    return 'Likely a Spam' if prediction == 1 else 'Likely Not a Spam'

# Define a dictionary mapping model names to their file paths
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

if st.button("Predict"):
    if uploaded_file is not None:
        # Read sentences from the uploaded CSV file
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

    # Define a function to apply color based on the prediction value
    def highlight_spam(val):
        color = 'darkred' if val == 'Likely a Spam' else 'darkgreen'
        return f'background-color: {color}'

    # Check if the predictions are based on a single sentence from the input box
    if len(sentences) == 1 and uploaded_file is None:
        # Display the dataframe without the 'Sentence' column
        st.dataframe(results_df.drop(columns=['Sentence']).style.applymap(highlight_spam, subset=['Prediction']))
    else:
        # Display the full dataframe
        st.dataframe(results_df.style.applymap(highlight_spam, subset=['Prediction']))
