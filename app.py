import streamlit as st
import pickle
import pandas as pd
import nltk
nltk.data.path.append('nltk_data')
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb')) 

st.title("Email Spam Classifier")
st.write("This app classifies emails as spam or not spam.")

email_text = st.text_area("Enter the email text here:")

# preprocessing
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y =[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    

    return ' '.join(y)

input = transform_text(email_text)
# vectorize
data = tfidf.transform([input])

if st.button("Classify"):
    if email_text:
        
        # Make prediction   
        prediction = model.predict(data)
        # prediction = model.predict(data['text'])

        if prediction[0] == 1:
            st.success("Spam.")
        else:
            st.success("Not Spam.")
    else:
        st.warning("Please enter some text to classify.")

st.markdown(
    """
    <style>
    .stApp {
        # background-image: url("https://images.unsplash.com/photo-1563832528262-15e2bca87584?q=80&w=2019&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-image: url("https://images.unsplash.com/photo-1478760329108-5c3ed9d495a0?q=80&w=1974&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        # opacity: 0.5;
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)