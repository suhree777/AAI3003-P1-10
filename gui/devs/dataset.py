import streamlit as st
from datasets import load_dataset
import pandas as pd

st.header("Dataset used")

st.markdown(
    """
    The dataset used is fixed and consistent for all training and testing the models.
    """
)
# Return the dataset for use in other files

def save_dataset():
    dataset = load_dataset("sms_spam")
    return dataset

# Load the dataset
dataset = save_dataset()

# Convert the dataset to a Pandas DataFrame
# Adjust the slice [:1000] to display a different number of rows
df = pd.DataFrame(dataset['train'][:1000])

# Display the DataFrame
st.dataframe(df)
