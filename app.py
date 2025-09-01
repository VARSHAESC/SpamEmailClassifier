import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("spam_classifier_model.pkl")        # Your trained ML model
vectorizer = joblib.load("vectorizer.pkl")        # Your TfidfVectorizer or CountVectorizer

# Streamlit app title
st.title("📧 Email Spam Classifier")
st.write("Enter your email content below to check if it's Spam or Not Spam.")

# User input
email_input = st.text_area("Email Content", height=200)

# Predict button
if st.button("Check Email"):
    if email_input.strip() == "":
        st.warning("Please enter an email message to classify.")
    else:
        # Transform the input using the saved vectorizer
        input_vector = vectorizer.transform([email_input])
        prediction = model.predict(input_vector)[0]

        # Display result
        if prediction == 1:
            st.error("⚠️ This email is classified as SPAM!")
        else:
            st.success("✅ This email is NOT Spam.")
