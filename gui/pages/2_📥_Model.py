import streamlit as st
import pickle

st.header("Model Upload")
# Allow users to upload a model
uploaded_model = st.file_uploader("Upload your spam detection model", type=["pkl", "h5"])

if uploaded_model is not None:
    # Save the uploaded model to session state for access in 3_📋_Evaluation.py
    st.session_state['uploaded_model'] = uploaded_model

    # Display a message to navigate to the evaluation page
    st.write("Model uploaded successfully. Please go to the", "<span style='color: blue;'>📋 Evaluation</span>", "page to see the performance analysis.", unsafe_allow_html=True)
else:
    st.write("Please upload a model to evaluate its performance.")
