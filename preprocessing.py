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

def clean_text(text: str) -> str:
    """
    Cleans text by converting to lowercase and removing punctuation and numbers.

    Args:
        text (str): The input string.

    Returns:
        str: The cleaned string, containing only lowercase letters and spaces.
    """
    if not isinstance(text, str):
        # Or raise TypeError, but for this project, returning empty string might be safer if called in a pipeline
        print(f"Warning: clean_text received non-string input: {type(text)}. Returning empty string.")
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) # Keep only letters and spaces
    return text

def tokenize_text(text: str) -> list[str]:
    """
    Tokenizes text into words using NLTK.

    Args:
        text (str): The input string.

    Returns:
        list[str]: A list of word tokens.
    """
    if not isinstance(text, str):
        print(f"Warning: tokenize_text received non-string input: {type(text)}. Returning empty list.")
        return []
    return nltk.word_tokenize(text)

def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Removes common English stopwords from a list of tokens.

    Args:
        tokens (list[str]): A list of word tokens.

    Returns:
        list[str]: A list of tokens with stopwords removed.
    """
    if not isinstance(tokens, list):
        print(f"Warning: remove_stopwords received non-list input: {type(tokens)}. Returning empty list.")
        return []
    return [word for word in tokens if word not in stop_words]

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """
    Lemmatizes tokens to their base form using WordNetLemmatizer.

    Args:
        tokens (list[str]): A list of word tokens.

    Returns:
        list[str]: A list of lemmatized tokens.
    """
    if not isinstance(tokens, list):
        print(f"Warning: lemmatize_tokens received non-list input: {type(tokens)}. Returning empty list.")
        return []
    return [lemmatizer.lemmatize(word) for word in tokens]

def preprocess_text(text: str, perform_lemmatization: bool = True) -> str:
    """
    Applies a full preprocessing pipeline to the input text.

    The pipeline consists of:
    1. Cleaning (lowercase, remove punctuation/numbers).
    2. Tokenization.
    3. Stopword removal.
    4. Lemmatization (optional).

    Args:
        text (str): The input string.
        perform_lemmatization (bool, optional): Whether to perform lemmatization.
                                                 Defaults to True.

    Returns:
        str: The preprocessed text as a single string with tokens joined by spaces.
             Returns an empty string if input is not a string.
    """
    if not isinstance(text, str):
        print(f"Warning: preprocess_text received non-string input: {type(text)}. Returning empty string.")
        return ""

    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    tokens = remove_stopwords(tokens)
    if perform_lemmatization:
        tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)

def extract_medical_entities_spacy(text: str) -> list[tuple[str, str]]:
    """
    Extracts named entities from text using the loaded spaCy model.

    Identifies entities like diseases, drugs, anatomical parts, etc.
    This function is crucial for identifying keywords for Wikipedia searches.

    Args:
        text (str): The input string.

    Returns:
        list[tuple[str, str]]: A list of tuples, where each tuple contains
                                the entity text and its label (e.g., ("diabetes", "DISEASE")).
                                Returns an empty list if input is not a string or if an error occurs.
    """
    if not isinstance(text, str):
        print(f"Warning: extract_medical_entities_spacy received non-string input: {type(text)}. Returning empty list.")
        return []
    try:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
    except Exception as e:
        print(f"Error during spaCy entity extraction for text snippet '{text[:50]}...': {e}")
        return []
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