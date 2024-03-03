import streamlit as st
from transformers import pipeline
import pickle

st.header("Model Upload")

# Allow users to upload a model
uploaded_models = st.file_uploader("Upload your spam detection models", type=["pkl", "h5", "safetensors"], accept_multiple_files=True)

def load_model(uploaded_model):
    if uploaded_model.type == 'application/octet-stream':
        model = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
    else:
        with open(uploaded_model.name, 'rb') as f:
            model = pickle.load(f)
    return model

def spam_or_ham(model):
    st.write('Predicting if message is spam or not')
    message = st.text_input('Enter a message')
    submit = st.button('Predict')

    if submit:
        prediction = model(message)[0]

        if prediction['label'] == '1':
            st.warning('This message is spam.')
        else:
            st.success('This message is Legit (HAM)')

if uploaded_models:
    # Initialize a dictionary to store uploaded models if it doesn't exist
    if 'models' not in st.session_state:
        st.session_state['models'] = {}

    for uploaded_model in uploaded_models:
        # Load the uploaded model
        model = load_model(uploaded_model)

        # Save the uploaded model to session state for access in 3_📋_Evaluation.py
        st.session_state['models'][uploaded_model.name] = model

        # Display a message to navigate to the evaluation page for each model
        st.write(f"Model '{uploaded_model.name}' uploaded successfully. Please go to the",
                 "<span style='color: blue;'>📋 Evaluation</span>", "page to see the performance analysis.", unsafe_allow_html=True)

        # Call the spam_or_ham function to make predictions
        spam_or_ham(model)

else:
    st.write("Please upload at least one model to evaluate its performance.")
