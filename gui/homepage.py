import streamlit as st
import os
from datasets import load_dataset
# Define your pages
pages = {
    "💯 Home": "homepage.py",
    "📥 Model": "2_📥_Model.py",
    "📋 Evaluation": "3_📋_Evaluation.py"
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

def handle_devs_button():
    # Center-align the button in the sidebar
    if "show_additional_pages" not in st.session_state:
        st.session_state.show_additional_pages = False

    if st.sidebar.button("For Devs"):
        st.session_state.show_additional_pages = not st.session_state.show_additional_pages

    # Show the content of dataset.py when "For Devs" button is clicked and the section is expanded
    if st.session_state.show_additional_pages:
        st.sidebar.header("Datasets")
        # Create a button or checkbox to load the dataset
        if st.sidebar.button("Load Dataset"):
            # Load and execute the content of dataset.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_path = os.path.join(current_dir, "devs", "dataset.py")
            with open(dataset_path, "r") as file:
                code = file.read()
            exec(code)

def main():
    set_page_config()
    display_title_and_description()
    handle_devs_button()

if __name__ == "__main__":
    main()
