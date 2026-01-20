import streamlit as st
import pickle
accuracy = pickle.load(open("accuracy.pkl", "rb"))

st.set_page_config(page_title="Spam Checker", layout="centered")

# Load model & vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📩 Spam Email Checker")
st.markdown(f"""
<div style="
    background-color:#1c1c1c;
    padding:12px 20px;
    border-radius:12px;
    display:inline-block;
    margin-bottom:20px;
">
    🎯 <b>Model Accuracy:</b> {accuracy*100:.2f}%
</div>
""", unsafe_allow_html=True)

st.write("Paste your email content below and check if it is spam or not.")

message = st.text_area("Email Content", height=200)

if st.button("Check Spam"):
    if message.strip() == "":
        st.warning("Please enter some text")
    else:
        input_data = cv.transform([message])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        spam_prob = probabilities[list(model.classes_).index("Spam")] * 100
        ham_prob = probabilities[list(model.classes_).index("Not Spam")] * 100

        if prediction == "Spam":
            st.error(f"🚨 This email is SPAM")
        else:
            st.success(f"✅ This email is NOT spam")

        st.markdown(f"""
        <div style="
            background-color:#1c1c1c;
            padding:16px;
            border-radius:14px;
            margin-top:15px;
        ">
            📊 <b>Prediction Confidence</b><br><br>
            🚨 Spam Probability: <b>{spam_prob:.2f}%</b><br>
            ✅ Not Spam Probability: <b>{ham_prob:.2f}%</b>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")
st.markdown("")

if st.button("⬅ Back to Home"):
    st.switch_page("spamdetection.py")
