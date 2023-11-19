# from flask import Flask , render_template , request
import streamlit as st
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# app = Flask(__name__)

#Bulid functionalities
# @app.route('/' , methods = ['GET'])
# def home():
#     return render_template('index.html')

news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author']+"  "+news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']


ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]'," ",content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content

news_df['content'] = news_df['content'].apply(stemming)

X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

#split the data set
X_train ,X_test ,y_train ,y_test = train_test_split(X , y, test_size = 0.2 , stratify = y ,random_state = 1)

#fit logisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


#website
st.title('Fake news detection')
input_text = st.text_input('Enter news Article :')


def prediction(input_text):
    input_data = vector.transform([input_text])
    predicition = model.predict(input_data)
    return predicition[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.write('The news is fake')
    else:
        st.write('News is real')