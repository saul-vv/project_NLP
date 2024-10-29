import streamlit as st
import pickle
from sklearn.svm import SVC

from nltk.corpus import wordnet

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

def get_wordnet_pos(word):
    """
    Map the results of pos_tag() to the characters that lemmatize() accepts
    """
    # from nltk.corpus import wordnet
    tag = nltk.pos_tag([word])[0][1][0]
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def data_cleaning(text):
    # import re
    # import nltk
    # from nltk.corpus import stopwords
    # from nltk.stem.wordnet import WordNetLemmatizer

    text = text.lower()
    text = re.sub(r'[^A-Za-z\s]+', ' ', text) # Regex to remove all the special characters and numbers
    text = re.sub(r'\b\w\b', '', text) # Regex to remove all single characters
    text = re.sub(r' {2,}', ' ', text).strip() # Regex to substitute multiple spaces with single space
    
    tokenized_text = nltk.word_tokenize(text)
    text = [WordNetLemmatizer().lemmatize(word, get_wordnet_pos(word)) for word in tokenized_text if word not in stopwords.words("english")]

    text = " ".join(text) # Transforms the list of words back into a single string
    return text
  

st.title("Reviews App")
st.write('This app will tell you if a customer review is positive, neutral or negative')

st.header("Type a review")
review = st.text_input("type your review here", "e.g.: This product is amazing!")

if st.button("Analyze review"):
    st.write("Analyzing review...")
    
    review = data_cleaning(review)
    review = pickle.load(open('vectorizer.pkl', 'rb')).transform([review])

    model = pickle.load(open('model_SVC.pkl', 'rb'))
    prediction = model.predict(review)

    st.write(prediction)
    