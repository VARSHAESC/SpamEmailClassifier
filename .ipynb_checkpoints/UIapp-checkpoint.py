import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page config
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# Title with style
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>📧 Spam Email Classifier</h1>",
    unsafe_allow_html=True
)

st.write("Enter an email or message below to check if it’s **Spam** or **Ham (Not Spam)**.")

# Input box
user_input = st.text_area("✍️ Your message:", height=150)

# Predict button
if st.button("🔍 Predict"):
    if user_input.strip() != "":
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]

        if prediction == 1:
            st.error("🚨 **This message is Spam!**")
        else:
            st.success("✅ **This message is Ham (Not Spam).**")
    else:
        st.warning("⚠️ Please enter a message!")

# Footer
st.markdown(
    "<hr><p style='text-align: center; color: grey;'>Built with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)

