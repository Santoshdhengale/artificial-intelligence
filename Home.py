import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Spam Email Detection",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f0f0f;
}

.main-title {
    font-size: 52px;
    font-weight: 700;
    text-align: center;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 40px;
}

.section-title {
    text-align: center;
    font-size: 28px;
    margin: 40px 0 30px 0;
}

.card {
    background-color: #1c1c1c;
    padding: 30px;
    border-radius: 16px;
    text-align: center;
    height: 220px;
}

.card h3 {
    margin-bottom: 15px;
}

.card p {
    color: #9ca3af;
    font-size: 16px;  
    line-height: 1.6; 
}


.big-box {
    background-color: #1c1c1c;
    padding: 40px;
    border-radius: 20px;
    margin-top: 50px;
}

.big-box ul {
    list-style: none;
    padding-left: 0;
}

.big-box li {
    margin-bottom: 12px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGO ----------------
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([10,1,10])

with col2:
    st.image("security2.png", width=150)

st.markdown("""
<p style="text-align:center; color:#9ca3af; letter-spacing:1px;">
STAY PROTECTED
</p>
""", unsafe_allow_html=True)



# ---------------- TITLE ----------------
st.markdown("<div class='main-title'>Spam Email Detection</div>", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Protect your inbox with our advanced AI-powered spam detection system
</div>
""", unsafe_allow_html=True)

# ---------------- HOW IT WORKS ----------------
st.markdown("<div class='section-title'>How it Works 🤔</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>1. Input Your Email 📧</h3>
        <p>Paste the email content you want to analyze into our secure text box</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>2. AI Analysis ⚡️</h3>
        <p>Our machine learning model analyzes patterns, keywords, and context</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3>3. Get Results ✅</h3>
        <p>Receive instant feedback with spam probability percentage</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- WHY CHOOSE US ----------------
st.markdown("""
<div class="big-box">
    <h2>Why Choose Our Spam Detector?</h2>
    <ul>
        <li>✅ <b>Accurate Detection:</b> Advanced AI algorithms trained on millions of emails</li>
        <li>✅ <b>Instant Results:</b> Get your spam analysis in seconds</li>
        <li>✅ <b>Privacy First:</b> Your emails are analyzed locally and never stored</li>
        <li>✅ <b>Easy to Use:</b> Simple interface designed for everyone</li>
    </ul>
</div>
""", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    if st.button("🚀 Let’s Go →", use_container_width=True):
        st.switch_page("pages/1_Spam_Checker.py")


#it is used to classify the text messages as spam or not spam based on the frequency of words in the messages.

data = pd.read_csv(r"C:\Users\dhane\Documents\Spam Detector\spam.csv", encoding='latin-1')

#data preprocessing and cleaning

data.drop_duplicates(inplace=True)
data['Category']= data['Category'].replace(['ham','spam'],['Not Spam','Spam'])

messages = data['Message']
cat = data['Category']

(messages_train, messages_test, cat_train, cat_test) = train_test_split(messages, cat, test_size=0.2)
# 0.2 means 20% of data will be used for testing and 80% for training and train_test_split is used to split the data into training and testing sets.

cv = CountVectorizer(stop_words='english')
#here we are using CountVectorizer to convert the text messages into a matrix of token counts, while also removing common English stop words such as a and the etc.

features_train = cv.fit_transform(messages_train)
features_test = cv.transform(messages_test)

#creating model
model = MultinomialNB()
model.fit(features_train, cat_train)

#testing the model

predictions = model.predict(features_test)
accuracy = (predictions == cat_test).mean()

#predicting new data
def predict(message):
    input_message = cv.transform([message]) 
    result = model.predict(input_message) 
    return "🚨 Spam" if result [0] =="Spam" else "✅ Not Spam"


#saving the model
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))
pickle.dump(accuracy, open("accuracy.pkl", "wb"))

