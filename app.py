import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Spam Email Classifier")

user_input = st.text_area("Enter an email/message:")

if st.button("Predict"):
    if user_input.strip() != "":
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]

        result = "🚨 Spam" if prediction == 1 else "✅ Ham (Not Spam)"
        st.write("### Result:", result)
    else:
        st.warning("Please enter a message!")
