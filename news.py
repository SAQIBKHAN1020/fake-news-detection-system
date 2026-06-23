import streamlit as st
import pandas as pd
import re
import nltk




from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


nltk.download("stopwords")

st.title("Fake News Detection System:")

# _________________ Pandas sa Data Loads_____________________

fake=pd.read_csv("Fake.csv")
true=pd.read_csv("True.csv")

fake = fake.head(500)
true = true.head(500)


# Labels
fake["label"]=0
true["label"]=1

# Machine learning ko batana hota:       # fake kya
                                                        # real kya

                                                                                # So:

                                                                                                # 0 = fake
                                                                                                            # 1 = real


news=pd.concat([fake,true],axis=0)         #Dono datasets merge. #axis=0   #👉 vertically   #👉 rows wis


news=news[["text","label"]]

news=news.sample(frac=1)

ps = PorterStemmer()

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = re.sub('[^a-zA-Z]', ' ', text)

    text = text.lower()

    text = text.split()

    text = [
        ps.stem(word)
        for word in text
        if word not in stop_words
    ]

    return " ".join(text)

news["text"]=news["text"].apply(clean_text)


X=news["text"]
y=news["label"]

vectorizer=TfidfVectorizer(max_features=5000)

X=vectorizer.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


model=LogisticRegression()

model.fit(X_train,y_train)

prediction=model.predict(X_test)
accuracy=accuracy_score(y_test,prediction)

st.subheader("Model Accuracy:")
st.success(f"Accuracy :  {accuracy * 100:.2f}%")

st.subheader("Check News:")

user_news=st.text_area("Enter the News Article")
if st.button("Detect News:"):
    cleaned_news=clean_text(user_news)
    vector_input=vectorizer.transform([cleaned_news])
    result=model.predict(vector_input)

    if result[0]==1:
        st.success("REAL NEWS:")

    else:
        st.error("FAKE NEWS")
