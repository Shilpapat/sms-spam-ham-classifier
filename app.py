import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("EMAIL/SMS SPAM CLASSIFIER")

input_sms = st.text_input("Enter the message")



# from here we need to work on 3 steps
# 1. preprocessing for this we are using the transform_text function

def transform_text(text):
    text = text.lower() # first step to convert into lower case
    text = nltk.word_tokenize(text) # second step is to convert the lower letter words into tokens..
    y = [] # this is to remove the special chracters..
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:] # Third step, this is nothing but cloning of the list in text..
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:] # this is forth step for stemming..
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

if st.button('Predict') :
    # 1. here we are calling above function to give output
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1 :
        st.header("SPAM")
    else :
        st.header("NOT SPAM")

