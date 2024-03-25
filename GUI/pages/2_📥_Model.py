import streamlit as st
import pickle
import pandas as pd
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from st_aggrid import AgGrid, GridUpdateMode, JsCode
import altair as alt

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [ps.stem(word) for word in text if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(text)

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

def get_filtered_data(grid_response):
    # Use the response from the AgGrid to get the filtered data
    return grid_response['data']

def plot_chart(data):
    # Calculate the proportion of sentences classified as spam by each model
    spam_proportions = data.groupby('Model')['Prediction'].apply(lambda x: (x == 'Likely a Spam').mean()).reset_index(name='Spam Proportion')
    
    # Create an Altair chart
    chart = alt.Chart(spam_proportions).mark_bar().encode(
        x='Model',
        y='Spam Proportion',
        color='Model',
        tooltip=['Model', 'Spam Proportion']
    ).properties(
        width=600,
        height=400
    )
    
    st.altair_chart(chart, use_container_width=True)



# Streamlit app
st.title("Spam Detection Demo")
st.markdown("Enter a sentence or upload a CSV file with sentences for spam detection.")

sentence = st.text_input("Enter a sentence:")
uploaded_file = st.file_uploader("Or upload a CSV file with sentences:", type=["csv"])

if st.button("Predict") or 'results_df' in st.session_state:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'Sentence' in df.columns:
            st.write("### Prediction Results:")
            sentences = df['Sentence'].tolist()
        else:
            st.error("CSV file must have a column named 'Sentence'")
            sentences = []
    else:
        sentences = [sentence] if sentence else []

    all_results = []
    for sent in sentences:
        results = predict_spam(sent)
        for result in results:
            result['Sentence'] = sent
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)
    st.session_state['results_df'] = results_df  # Save results in session state

    # Define custom cell style based on the prediction
    cell_style_jscode = JsCode("""
    function(params) {
        if (params.value === 'Likely a Spam') {
            return {'color': 'white', 'backgroundColor': 'darkred'};
        } else {
            return {'color': 'white', 'backgroundColor': 'darkgreen'};
        }
    };
    """)

    # Configure the AgGrid component
    grid_options = {
        'columnDefs': [
            {'field': 'Sentence', 'filter': 'agTextColumnFilter', 'sortable': True, 'resizable': True},
            {'field': 'Model', 'filter': 'agTextColumnFilter', 'sortable': True, 'resizable': True},
            {'field': 'Prediction', 'cellStyle': cell_style_jscode, 'filter': 'agTextColumnFilter', 'sortable': True, 'resizable': True}
        ],
        'defaultColDef': {
            'editable': False,
            'filter': True,
            'sortable': True,
            'resizable': True
        }
    }

    # # Display the interactive dataframe for predictions
    grid_response = AgGrid(
        results_df, 
        gridOptions=grid_options, 
        update_mode=GridUpdateMode.VALUE_CHANGED, 
        fit_columns_on_grid_load=True, 
        theme='streamlit', 
        allow_unsafe_jscode=True,
        data_return_mode='FILTERED',
        key='predictions_grid'
    )

    # Plot the chart with the initial data
    plot_chart(results_df)

# Clear session state when the app is reloaded
if st.button("Clear Results"):
    if 'results_df' in st.session_state:
        del st.session_state['results_df']