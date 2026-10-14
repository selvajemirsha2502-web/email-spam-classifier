
import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

MODEL_PATH = "models/spam_model.pkl"

st.title("📧 Email Spam Classifier")
st.write("Check whether a message is Spam or Normal.")

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Please run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)

message = st.text_area(
    "Enter your message here:",
    placeholder="Type or paste your message...",
    height=180
)

if st.button("🔍 Check Message", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")
            st.write("This message may be unwanted or suspicious.")
        else:
            st.success("✅ NORMAL MESSAGE")
            st.write("This message looks like a normal message.")

st.caption(
    "Educational prototype trained on the SMS Spam Collection dataset."
)
