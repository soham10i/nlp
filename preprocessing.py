import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import spacy
import os
from config import SPACY_MODEL_NAME, SPACY_MODEL_PATH

# Download NLTK resources (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Load spaCy model (ensure it's downloaded: python -m spacy download en_core_web_sm)
try:
    if SPACY_MODEL_PATH and os.path.exists(SPACY_MODEL_PATH):
        nlp = spacy.load(SPACY_MODEL_PATH)
    else:
        nlp = spacy.load(SPACY_MODEL_NAME) 
except OSError:
    print(f"SpaCy model '{SPACY_MODEL_NAME}' not found locally. Downloading...")
    spacy.cli.download(SPACY_MODEL_NAME)
    nlp = spacy.load(SPACY_MODEL_NAME)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Removes punctuation, numbers, and converts to lowercase."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces
    return text

def tokenize_text(text):
    """Tokenizes text into words."""
    return nltk.word_tokenize(text)

def remove_stopwords(tokens):
    """Removes common stopwords from a list of tokens."""
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens):
    """Lemmatizes tokens to their base form."""
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text, perform_lemmatization=True):
    """
    Applies a full preprocessing pipeline.
    """
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    if perform_lemmatization:
        tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)

def extract_medical_entities_spacy(text):
    """
    Extracts named entities from text using spaCy. [cite: 140]
    This function is crucial for identifying keywords for Wikipedia searches. [cite: 186]
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

if __name__ == "__main__":
    sample_text = "What is the primary treatment for type 2 diabetes mellitus, considering patient history and medication side effects?"
    print(f"Original text: {sample_text}")

    processed_text = preprocess_text(sample_text)
    print(f"Processed text: {processed_text}")

    entities = extract_medical_entities_spacy(sample_text)
    print(f"Extracted Entities (spaCy): {entities}")

    wiki_content = "Diabetes mellitus (DM), commonly known as diabetes, is a metabolic disease that causes high blood sugar. The hormone insulin moves sugar from the blood into your cells to be stored or used for energy."
    processed_wiki_content = preprocess_text(wiki_content)
    print(f"\nProcessed Wikipedia Content: {processed_wiki_content}")