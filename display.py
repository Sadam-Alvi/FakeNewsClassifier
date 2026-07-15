import streamlit as st
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (run once)
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk_resources()

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    with open("best_svc_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

def preprocess_alphanumeric(text):
    """Preprocess text for the model."""
    if not isinstance(text, str) or not text.strip():
        return "", []
    
    # Remove non-alphanumeric characters except spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stop word removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    cleaned_text = ' '.join(tokens)
    return cleaned_text, tokens

# Streamlit UI
st.title("News Reliability Classifier")
st.markdown("Enter a news article or text snippet to determine its reliability.")

text_input = st.text_area(
    "Enter your text:", 
    height=200,
    placeholder="Paste the news content here..."
)

if st.button("Classify News"):
    if not text_input or not text_input.strip():
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Processing..."):
            # Preprocess
            cleaned_text, _ = preprocess_alphanumeric(text_input)
            
            if not cleaned_text:
                st.error("Unable to process the provided text.")
            else:
                # Transform using vectorizer
                # Note: Most vectorizers (e.g., TfidfVectorizer) expect a list/iterable of strings
                X_transformed = vectorizer.transform([cleaned_text])
                
                # Predict
                prediction = model.predict(X_transformed)[0]
                
                # Display result
                if prediction == 0:
                    st.success("✅ The news is classified as **Reliable**.")
                else:
                    st.error("❌ The news is classified as **Not Reliable**.")
                
                # Optional: Show confidence/probability if model supports it
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_transformed)[0]
                    st.info(f"Confidence: Reliable = {proba[0]:.2%}, Not Reliable = {proba[1]:.2%}")

# Sidebar information
st.sidebar.header("About")
st.sidebar.info(
    "This application uses a Support Vector Classifier (SVC) trained on news data.\n\n"
    "The text is preprocessed (cleaned, tokenized, stop words removed, and lemmatized) "
    "before being passed to the model."
)