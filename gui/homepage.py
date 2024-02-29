# pip install -U streamlit
# pip install -U plotly

# you can run your app with: streamlit run app.py

import streamlit as st
import pickle
import os

# Set page configuration
st.set_page_config(page_title="AAI1001", layout="wide", page_icon="💯")

# Center-align the button in the sidebar
if "show_additional_pages" not in st.session_state:
    st.session_state.show_additional_pages = False

if st.sidebar.button("For Devs"):
    st.session_state.show_additional_pages = not st.session_state.show_additional_pages

st.title("AAI1001 Data Engineering and Visualization Project")
st.header("Cardiovascular Diseases Prediction via Electrocardiogram")

st.markdown(
    """
    Done by:\n
    👧 LEO EN QI VALERIE                [2202795]\n
    👧 TEO XUANTING                     [2202217]\n
    TIAN YUE XIAO, BRYON                [2201615]\n
    👧SERI HANZALAH BTE HANIFFAH        [2201601]
    """
)

st.markdown(
    """
    This project aims to design a minimal viable product (MVP) 
    of a trained Machine Learning (ML) model with a Graphical User Interface (GUI) 
    to predict heart disease. It addresses the need for accurate detection, 
    objective assessments, and efficient usage of resources in diagnosing heart disease.
    """
)

# Show the content of 2_Normal_Evaluation.py and 3_Dataframe.py when "For Devs" button is clicked
if st.session_state.show_additional_pages:
    st.sidebar.header("Normal Evaluation")

    # Load and execute the content of 2_Normal_Evaluation.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    normal_evaluation_path = os.path.join(current_dir, "devs", "2_Normal_Evaluation.py")
    with open(normal_evaluation_path, "r") as file:
        code = file.read()
    exec(code)


# loading the trained model
model = pickle.load(open('model.pkl', 'rb'))

# create title
st.title('Predicting if message is spam or not')

message = st.text_input('Enter a message')

submit = st.button('Predict')

if submit:
    prediction = model.predict([message])

    # print(prediction)
    # st.write(prediction)
    
    if prediction[0] == 'spam':
        st.warning('This message is spam')
    else:
        st.success('This message is Legit (HAM)')


st.balloons()