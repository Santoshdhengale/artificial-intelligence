import pandas as pd
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📧",
    layout="centered"
)

# -------------------- CUSTOM CSS (FIGMA-STYLE UI) --------------------
st.markdown("""
<style>
textarea {
    border-radius: 12px !important;
}

div.stButton > button {
    background-color: #22c55e;
    color: black;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    font-weight: bold;
}

.sidebar .sidebar-content {
    background-color: #f8fafc;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD DATA (USE RELATIVE PATH) --------------------
data = pd.read_csv("spam.csv")   # IMPORTANT: relative path for GitHub

data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

mess = data['Message']
cat = data['Category']


# -------------------- TRAIN MODEL (ONCE) --------------------
mess_train, mess_test, cat_train, cat_test = train_test_split(
    mess, cat, test_size=0.2, random_state=42
)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

model = MultinomialNB()
model.fit(features, cat_train)


# -------------------- PREDICTION FUNCTION --------------------
def predict(message):
    input_message = cv.transform([message])
    result = model.predict(input_message)
    return "🚨 Spam" if result[0] == "Spam" else "✅ Not Spam"


# -------------------- SIDEBAR NAVIGATION --------------------
# -------------------- PAGE STATE --------------------
if "page" not in st.session_state:
    st.session_state.page = "home"


# -------------------- EXTRA CSS FOR CARDS + BUTTON --------------------
st.markdown("""
<style>
.card-container {
    display: flex;
    gap: 24px;
    margin-top: 40px;
}

.card {
    background: linear-gradient(180deg, #0f172a, #020617);
    color: #ffffff;
    border-radius: 20px;
    padding: 30px;
    width: 100%;
    min-height: 220px;
    text-align: center;
    box-shadow: 0px 12px 40px rgba(0,0,0,0.6);
}

.wide-card {
    background: linear-gradient(180deg, #0f172a, #020617);
    color: #ffffff;
    border-radius: 20px;
    padding: 36px;
    margin-top: 50px;
    box-shadow: 0px 12px 40px rgba(0,0,0,0.6);
}



.bottom-right {
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 999;
}

</style>
            
            
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* FORCE dark cards inside Streamlit markdown */
div[data-testid="stMarkdownContainer"] .card,
div[data-testid="stMarkdownContainer"] .wide-card {
    background: linear-gradient(180deg, #0f172a, #020617) !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ==================== PAGE 1: HOME ====================
if st.session_state.page == "home":

    # LOGO (you can replace logo.png)
    st.image(
        "https://cdn-icons-png.flaticon.com/512/561/561127.png",
        width=90
    )

    st.markdown("<h1 style='text-align:center;'>Spam Detector</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;'>AI-powered protection against unwanted messages</p>",
        unsafe_allow_html=True
    )

    # -------- 3 FEATURE BOXES --------
    st.markdown("""
<div class="card-container">
    <div class="card">
        <h2>🤖</h2>
        <h3>AI Powered</h3>
        <p>Advanced ML models trained on real-world spam data.</p>
    </div>

    <div class="card">
        <h2>⚡</h2>
        <h3>Fast Results</h3>
        <p>Instant predictions with optimized algorithms.</p>
    </div>

    <div class="card">
        <h2>🔒</h2>
        <h3>Privacy First</h3>
        <p>Your messages are processed locally and never stored.</p>
    </div>
</div>
""", unsafe_allow_html=True)


    # -------- WHY CHOOSE US BOX --------
    st.markdown("""
    <div class="wide-card">
    <h3>Why choose our Spam Detector?</h3>
    <ul>
        <li><b>Accurate Detection:</b> Advanced AI algorithms trained on millions of emails.</li>
        <li><b>Instant Results:</b> Get your spam analysis in seconds.</li>
        <li><b>Privacy First:</b> Your emails are analyzed locally and never stored.</li>
        <li><b>Easy to Use:</b> Simple interface designed for everyone.</li>
    </ul>
</div>
    """, unsafe_allow_html=True)

    # -------- BOTTOM RIGHT BUTTON --------
    st.markdown('<div class="bottom-right">', unsafe_allow_html=True)
    if st.button("🚀 Let’s Goo"):
        st.session_state.page = "checker"
    st.markdown('</div>', unsafe_allow_html=True)



# ==================== PAGE 2: SPAM CHECKER ====================
# ==================== PAGE 2: SPAM CHECKER ====================
if st.session_state.page == "checker":

    st.markdown("<div class='wide-card'>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align:center;'>Check Your Message</h2>", unsafe_allow_html=True)

    input_mess = st.text_area(
        "Enter your message below",
        height=150,
        placeholder="Type or paste the message here..."
    )

    if st.button("Predict"):
        if input_mess.strip() == "":
            st.warning("Please enter a message")
        else:
            output = predict(input_mess)
            if "Spam" in output:
                st.error(output)
            else:
                st.success(output)

    st.markdown("</div>", unsafe_allow_html=True)


