import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
#it is used to classify the text messages as spam or not spam based on the frequency of words in the messages.

data = pd.read_csv(r"C:\Users\dhane\Documents\Spam Detector\spam.csv")

#data preprocessing and cleaning

data.drop_duplicates(inplace=True)
data['Category']= data['Category'].replace(['ham','spam'],['Not Spam','Spam'])

mess = data['Message']
cat = data['Category']

(mess_train,mess_test,cat_train,cat_test) = train_test_split(mess,cat,test_size=0.2)
# 0.2 means 20% of data will be used for testing and 80% for training and train_test_split is used to split the data into training and testing sets.

cv = CountVectorizer(stop_words='english')
#here we are using CountVectorizer to convert the text messages into a matrix of token counts, while also removing common English stop words such as a and the etc.
 
features = cv.fit_transform(mess_train)

#creating model
model = MultinomialNB()
model.fit(features,cat_train)

#testing the model

features_test = cv.transform(mess_test)

#predicting new data
def predict(message):
    input_message = cv.transform([message]) 
    result = model.predict(input_message) 
    return "🚨 Spam" if result [0] =="Spam" else "✅ Not Spam"

st.header('Spam Email/Message Detector')
#example usage

input_mess = st.text_input('Enter your message here:')
if st.button("Predict"):
    output = predict(input_mess)   # <-- output CREATED here
    st.markdown(f"### Result: **{output}**")

#saving the model
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))


