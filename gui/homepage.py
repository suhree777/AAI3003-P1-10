
import streamlit as st
import os
from datasets import load_dataset
# Define your pages
pages = {
    "💯 Home": "homepage.py",
    "📋 Evaluation": "2_📋_Evaluation.py",
    "📋 Demo": "demo.py"
}

def set_page_config():
    # Set the page configuration
    st.set_page_config(page_title="AAI3003", layout="wide", page_icon="💯")

def display_title_and_description():
    # Display title and description
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

def main():
    set_page_config()
    display_title_and_description()

if __name__ == "__main__":
    main()
