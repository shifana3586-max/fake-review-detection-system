import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Dataset Load
df = pd.read_csv(r"C:\Users\academytraining\Desktop\fake review\fake reviews dataset.csv")

# Label Encoding
df['label'] = df['label'].map({'CG':1, 'OR':0})

# Text Cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['clean_text'] = df['text_'].apply(clean_text)

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1,2)
)
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Model Training
from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(X_train, y_train)

# UI
st.title("Fake Review Detection System")

review = st.text_area("Enter Review")


if st.button("Predict"):

    review_clean = clean_text(review)

    review_tfidf = tfidf.transform([review_clean])

    prediction = model.predict(review_tfidf)
    st.write("Prediction Value:", prediction[0])


    if prediction[0] == 1:
        st.error("Fake Review")
    else:
        st.success("Genuine Review")