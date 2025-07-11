import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Download NLTK stopwords
nltk.download('stopwords')

# Load and clean dataset
df = pd.read_csv("flipkart_data.csv")
df = df[['review', 'rating']].dropna()

# Preprocessing setup
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Clean dataset
df['cleaned_review'] = df['review'].apply(preprocess_text)
df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 4 else 0)

# Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

# Train models
nb_model = MultinomialNB(alpha=0.3)
nb_model.fit(X, y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Flipkart Review Sentiment Analyzer", layout="centered")

# 🎨 Custom Background Style
st.markdown("""
    <style>
    .stApp {
        background-color: #E8F0FE;
        color: #000000;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextArea, .stSelectbox, .stButton {
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Interface ----------
st.title("🛒 Flipkart Product Review Sentiment Analyzer")
st.write("Paste a Flipkart product review, select a model, and check the sentiment.")

# Input area
user_input = st.text_area("✍️ Enter Review Here", "")

# Model selection
model_choice = st.selectbox("🧠 Choose Prediction Model:", ["Naive Bayes", "Random Forest"])

# Predict button
if st.button("🔍 Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        cleaned = preprocess_text(user_input)
        vectorized = vectorizer.transform([cleaned])

        if model_choice == "Naive Bayes":
            prediction = nb_model.predict(vectorized)[0]
            prob = nb_model.predict_proba(vectorized)[0][prediction]
        else:
            prediction = rf_model.predict(vectorized)[0]
            prob = rf_model.predict_proba(vectorized)[0][prediction]

        # Show preprocessed review
        with st.expander("🧹 Cleaned Review Text"):
            st.code(cleaned)

        # Result output
        st.subheader("📊 Sentiment Prediction:")
        if prediction == 1:
            st.success(f"👍 Positive Review ({prob:.2%} confidence)")
        else:
            st.error(f"👎 Negative Review ({prob:.2%} confidence)")

# Footer
st.markdown("---")
st.caption("🔍 Built with  using Streamlit and scikit-learn")
