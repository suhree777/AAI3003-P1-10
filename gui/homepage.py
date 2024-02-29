import streamlit as st
import pickle
import os

# Set page configuration
st.set_page_config(page_title="AAI3003", layout="wide", page_icon="💯")

# Center-align the button in the sidebar
if "show_additional_pages" not in st.session_state:
    st.session_state.show_additional_pages = False

if st.sidebar.button("For Devs"):
    st.session_state.show_additional_pages = not st.session_state.show_additional_pages

st.title("AAI3003 Natural Language Processing (NLP)")
st.header("Spam Detection Analysis using Natural Language Processing")

st.markdown(
    """
    Done by:\n
    👧 LEO EN QI VALERIE                [2202795]\n
    👧 TEO XUANTING                     [2202217]\n
    👦 TIAN YUE XIAO, BRYON             [2201615]\n
    👧 SERI HANZALAH BTE HANIFFAH       [2201601]
    """
)

st.markdown(
    """
    This project aims to conduct a performance analysis on Spam Detection through Natural Language Processing.
    It addresses the need for accurate detection, objective assessments, and efficient usage of resources in spam detection.
    """
)

# Show the content of 2_Normal_Evaluation.py and 3_Dataframe.py when "For Devs" button is clicked
if st.session_state.show_additional_pages:
    st.sidebar.header("Datasets")

    # Load and execute the content of 2_Normal_Evaluation.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    normal_evaluation_path = os.path.join(current_dir, "devs", "dataset.py")
    with open(normal_evaluation_path, "r") as file:
        code = file.read()
    exec(code)