import streamlit as st

import pickle

# natural language toolkit
import nltk 
nltk.download('wordnet')
# Text cleaning
import string

# Text pre-processing
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# load models
lemm_text=r'lemmatization_tokenizer.pkl'
model = r'SVCtfidf.pkl'
voc = r'tfidf.pkl'

# load stop words list and extend list with selected words
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
list_stop_words = list(stop_words)
list_stop_words.extend(('said','trump','reuters','president','state','government','new','states',
                        'house','republican','people','united','told','party','washington','election',
                        'year','campaign','donald','clinton','(reuters)','govern','news','united', 'states','said'))

#define lemmatization function
lemmatizer = WordNetLemmatizer()
def my_lemmatization_tokenizer(text):
    
    for word in text:
        listofwords = text.split(' ')
        
    listoflemmatized_words = []
    
    
    for word in listofwords:
        if (not word in list_stop_words) and (word != ''):
            lemmatized_word = lemmatizer.lemmatize(word)
            listoflemmatized_words.append(lemmatized_word)
            
    return listoflemmatized_words


# set headline and text for entering text
st.write(" # NEWS CLASSIFIER: FAKE OR FACT?")
text = st.text_area("PLEASE ENTER THE NEWS ARTICLE YOU WOULD LIKE TO EVALUATE HERE:")

if st.button("PREDICT"):
    #load and open lemmatization token
    token = pickle.load(open(lemm_text,"rb"))
    #load and open tfidf vectorizer
    news_vectorizer = pickle.load(open(voc, "rb"))
    #transform text using tfidf vectorizer
    vect_text = news_vectorizer.transform([text]).toarray()
    #load and open SVC model
    SVC = pickle.load(open(model, "rb"))
    #predict using model
    prediction = SVC.predict(vect_text)
    
    if prediction == 1:
        st.write("This is a fact-based article.")
    else:
        st.write("This is a fake news article.")
