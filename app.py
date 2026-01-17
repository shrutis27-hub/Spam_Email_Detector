import streamlit as st
import joblib

# ---------- Background Color ----------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e8f0fe;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title(" Spam Email Detector")

message = st.text_area("Enter email text")

if st.button("Check"):
    if message.strip() == "":
        st.warning("Please enter some text")
    else:
        # Vectorize input
        message_vec = vectorizer.transform([message])

        # Predict
        prediction = model.predict(message_vec)

        if prediction[0] == 1:
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is NOT Spam (Ham)")
