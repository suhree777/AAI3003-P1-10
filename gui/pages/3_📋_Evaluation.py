import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# Set page configuration
st.set_page_config(page_title="AAI3003", layout="wide", page_icon="📋")

def main():
    st.header("Model Evaluation")
    # Check if the performance metrics are available
    if 'model_performance' in st.session_state:
        # Display the performance evaluation analysis
        st.write("### Performance Evaluation Analysis")
        st.write(st.session_state['model_performance'])
    else:
        # Display a message asking the user to upload a model in the 2_📥_Model.py page
        st.write("#### Please upload at least one model in", "<span style='color: red;'>📥 Model</span>.", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
