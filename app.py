#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import joblib

import nltk
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords

import base64
import json


def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"

# CSS to style the page


nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


# Function to map NLTK's part-of-speech tags to wordnet tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN


# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Set of English stop words
stop_words = set(stopwords.words('english'))


def nltk_lemmatizer(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tags
        if word.isalpha() and word.lower() not in stop_words
    ]
    return lemmas


@st.cache_resource()
def load_resources():
    try:
        column_transformer = joblib.load("JobLibs/column_transformer.joblib")
        label_encoder = joblib.load("JobLibs/label_encoder.joblib")

        # models = {name: joblib.load(f"JobLibs/{name}_best_model.joblib") for name in ['rf', 'lr', 'dt', 'svc', 'xgb']}
        models = {
            name: joblib.load(f"JobLibs/{name}_best_model.joblib") if name not in ['lr', 'xgb'] else joblib.load(
                f"JobLibs/{name}_baseline.joblib")
            for name in ['rf', 'lr', 'dt', 'svc', 'xgb']
        }
        with open('model_precisions.json', 'r') as f:
            precisions = json.load(f)
    except Exception as e:
        st.error(f"Failed to load models or transformers: {str(e)}")
        return None, None
    return (column_transformer, label_encoder, models), precisions


# Function to classify text
def classify_text(input_text, resources):
    column_transformer, label_encoder, models = resources
    df = pd.DataFrame([input_text], columns=['selftext'])
    X_transformed = column_transformer.transform(df)
    predictions = {model_name: label_encoder.inverse_transform(model.predict(X_transformed))[0] for model_name, model in models.items()}
    return predictions


def format_prediction(prediction):
    if prediction == 'ytj':
        return f'<span style="color:green;">Yes 👍</span>'
    elif prediction == 'ntj':
        return f'<span style="color:red;">No 👎</span>'


def main():
    set_page_styles()
    # Add logo
    st.image("subreddit.jpeg", use_column_width=True)
    st.markdown('<div class="header-background"></div>', unsafe_allow_html=True)
    # Add logo with specified height using CSS class
    st.markdown('<h1 style="color: black;">Jerk Prediction App</h1>', unsafe_allow_html=True)
    input_text = st.text_area("Enter your text here:", height=200)
    resources, precisions = load_resources()

    if resources:
        # column_transformer, label_encoder, models = resources
        model_names = {
            'rf': 'Random Forest',
            'lr': 'Logistic Regression',
            'dt': 'Decision Tree',
            'svc': 'Support Vector Classifier',
            'xgb': 'XG Boost'
        }

        if st.button('Predict'):
            predictions = classify_text(input_text, resources)
            data = [{'Model': model_names[name], 'Prediction': format_prediction(pred),
                     'Precision': precisions.get(name, "N/A")} for name, pred in predictions.items()]
            df_results = pd.DataFrame(data)
            st.write(create_html_table(df_results), unsafe_allow_html=True)


def create_html_table(df):
    # Generate HTML for the table with proper styling
    table_html = df.to_html(index=False, escape=False, classes='custom-table')
    return f'<div class="container">{table_html}</div>'


def set_page_styles():
    st.markdown("""
    <style>
    .stApp {
        background-color: lavender;
    }
    .container {
        width: 100%;
        max-width: 800px;
        margin: auto;
    }
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }
    .custom-table th {
        background-color: #FF8C00; /* Darker orange background for header */
        color: white; /* White text color for header */
        font-weight: bold;
        padding: 10px;; /* Orange background for header */
        color: white; /* White text color for header */
        font-weight: bold;
        padding: 10px;
        text-align: center;
    }
    .custom-table td {
        border: 1px solid #ddd;
        background-color: #f8f8f8; /* Light grey background for rows */
        color: black;
        padding: 10px;
        text-align: center;
    }
    .custom-table tr:hover td {
        background-color: #e0e0e0; /* Darker grey background on hover */
    }
    </style>
    """, unsafe_allow_html=True)


# Set page configuration
st.set_page_config(page_title="Jerk Prediction", page_icon=":smiley:", layout="centered")


if __name__ == '__main__':
    main()
