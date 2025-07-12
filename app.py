import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Safe way to download only if needed
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

ensure_nltk_data()


# -------------------------
# 📌 Preprocessing Function
# -------------------------
ps = PorterStemmer()

def transform_text(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Keep alphanumeric
    tokens = [word for word in tokens if word.isalnum()]
    # Remove stopwords and punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Stemming
    tokens = [ps.stem(word) for word in tokens]
    return " ".join(tokens)

# -------------------------
# 🔍 Load Model and Vectorizer
# -------------------------
@st.cache_resource
def load_model():
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return vectorizer, model

tfidf, model = load_model()

# -------------------------
# 🚀 Streamlit App Interface
# -------------------------
st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("📩 Email/SMS Spam Classifier")
st.write("Enter your message below and find out if it's spam or not!")

# Input field
input_sms = st.text_area("✉️ Your message:")

# Predict button
if st.button('🔍 Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        # Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # Predict
        result = model.predict(vector_input)[0]

        # Output
        st.subheader("Prediction:")
        if result == 1:
            st.error("🔴 This message is **Spam**.")
        else:
            st.success("🟢 This message is **Not Spam**.")

        # Show the preprocessed version (optional)
        with st.expander("🔎 See Preprocessed Text"):
            st.code(transformed_sms)

