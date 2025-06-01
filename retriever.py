import os
import json
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # Added import
from config import SENTENCE_TRANSFORMER_MODEL, SENTENCE_TRANSFORMER_MODEL_PATH, \
                   WIKIPEDIA_TITLE_EMBEDDINGS_PATH, WIKIPEDIA_TITLE_INDEX_PATH, \
                   WIKIPEDIA_TITLE_METADATA_PATH, DEVICE, WIKIPEDIA_SEARCH_RESULTS_TO_CONSIDER, \
                   MAX_WIKIPEDIA_PAGES_PER_QUESTION, WIKIPEDIA_CACHE_DIR, QA_MODEL_NAME, QA_MODEL_PATH
from preprocessing import preprocess_text, extract_medical_entities_spacy

# Define relevant entity types for query augmentation
RELEVANT_ENTITY_TYPES = ["DISEASE", "DRUG", "ANATOMY", "PATHOLOGY", "PROCEDURE", "CHEMICAL"]

class WikipediaRetriever:
    def __init__(self):
        """
        Initializes the WikipediaRetriever.

        Loads the Sentence Transformer model for embeddings and ensures
        the Wikipedia cache directory exists.
        """
        print(f"Loading Sentence Transformer model to {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER_MODEL_PATH)
        self.model = AutoModel.from_pretrained(SENTENCE_TRANSFORMER_MODEL_PATH).to(DEVICE)
        self.model.eval() # Set model to evaluation mode

        self.faiss_index = None
        self.wiki_title_metadata = [] # List of {"title": ..., "pageid": ..., "summary": ...}

        # Ensure cache directory exists
        os.makedirs(WIKIPEDIA_CACHE_DIR, exist_ok=True)

    def _mean_pooling(self, model_output: tuple, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs mean pooling on token embeddings to get sentence embeddings.

        Args:
            model_output (tuple): Output from the Hugging Face model, where the first element
                                  (model_output[0]) contains token embeddings.
            attention_mask (torch.Tensor): Attention mask for the input tokens.

        Returns:
            torch.Tensor: Sentence embeddings.
        """
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts using the Sentence Transformer model.

        Args:
            texts (list[str]): A list of text strings to embed.

        Returns:
            np.ndarray: A NumPy array of sentence embeddings.
        """
        if not texts or not isinstance(texts, list):
            return np.array([])

        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings.cpu().numpy()

    def _get_wikipedia_search_results(self, query: str) -> list[dict]:
        """
        Performs an online search using the Wikipedia API for page titles and snippets.

        Args:
            query (str): The search query string.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary contains 'title',
                        'pageid', and 'snippet' of a search result. Returns an empty
                        list if the search fails or yields no results.
        """
        S = requests.Session()
        URL = "https://en.wikipedia.org/w/api.php"
        PARAMS = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srprop": "snippet", # Get a short summary
            "srlimit": WIKIPEDIA_SEARCH_RESULTS_TO_CONSIDER # Number of search results to consider
        }

        try:
            R = S.get(url=URL, params=PARAMS)
            R.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            DATA = R.json()
            if "query" in DATA and "search" in DATA["query"]:
                results = []
                for s in DATA["query"]["search"]:
                    results.append({
                        "title": s["title"],
                        "pageid": s["pageid"],
                        "snippet": s.get("snippet", "")
                    })
                return results
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error during Wikipedia API search for query '{query}': {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from Wikipedia API for query '{query}': {e}")
            return []


    def _get_wikipedia_page_content(self, title: str) -> str:
        """
        Downloads the full plain text content of a Wikipedia page by its title.

        The content is cached locally in the `WIKIPEDIA_CACHE_DIR` to avoid
        re-downloading on subsequent requests for the same page.

        Args:
            title (str): The title of the Wikipedia page.

        Returns:
            str: The plain text content of the page. Returns an empty string if
                 the page cannot be fetched or content is not available.
        """
        if not isinstance(title, str) or not title.strip():
            print("Warning: _get_wikipedia_page_content received invalid title. Returning empty string.")
            return ""

        # Check cache first
        # Sanitize title for use as a filename
        safe_filename = title.replace(' ', '_').replace('/', '_').replace('\\', '_')
        # Limit filename length to avoid OS errors
        max_filename_len = 200 # Conservative limit
        safe_filename = safe_filename[:max_filename_len]

        cached_filepath = os.path.join(WIKIPEDIA_CACHE_DIR, f"{safe_filename}.txt")
        if os.path.exists(cached_filepath):
            with open(cached_filepath, 'r', encoding='utf-8') as f:
                return f.read()

        S = requests.Session()
        URL = "https://en.wikipedia.org/w/api.php"
        PARAMS = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "titles": title,
            "explaintext": True, # Get plain text
            "exlimit": 1 # Only one extract
        }

        try:
            R = S.get(url=URL, params=PARAMS)
            R.raise_for_status()
            DATA = R.json()
            page_content = ""
            if "query" in DATA and "pages" in DATA["query"]:
                page_id = list(DATA["query"]["pages"].keys())[0]
                if page_id != "-1": # Page found
                    page_content = DATA["query"]["pages"][page_id].get("extract", "")
                    # Cache content
                    with open(cached_filepath, 'w', encoding='utf-8') as f:
                        f.write(page_content)
            return page_content
        except requests.exceptions.RequestException as e:
            print(f"Error during Wikipedia API content fetch for title '{title}': {e}")
            return ""
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from Wikipedia API for title '{title}': {e}")
            return ""

    def build_title_index(self):
        """
        This function is conceptual. In a real scenario, building a comprehensive
        title/summary index would involve scraping a large number of pages
        or using a pre-existing dataset of Wikipedia titles/summaries.
        For this project, we might initially just search and embed search results dynamically.
        However, if you want a local FAISS index for faster title selection from a fixed set,
        you'd need to gather titles/summaries beforehand.
        """
        print("Building a conceptual Wikipedia title index (requires a large-scale data collection)...")
        # For the purpose of this project, we assume this index will be built
        # from a pre-defined list of relevant Wikipedia titles/summaries OR
        # by iteratively caching results from common medical searches.
        # This will be a significant one-time setup cost.
        
        # Example: Fetching some top medical pages for index building
        # This is a simplification; a real index would be much larger.
        initial_search_terms = ["Medicine", "Disease", "Pharmacology", "Anatomy", "Pathology", "Surgery"]
        all_titles_data = []
        seen_pageids = set()

        for term in initial_search_terms:
            search_results = self._get_wikipedia_search_results(term)
            for res in search_results:
                if res["pageid"] not in seen_pageids:
                    all_titles_data.append(res)
                    seen_pageids.add(res["pageid"])
        
        if not all_titles_data:
            print("No data collected for title index. Cannot build FAISS index for titles.")
            return

        titles_and_snippets = [f"{item['title']} - {item['snippet']}" for item in all_titles_data]
        embeddings = self.get_embeddings(titles_and_snippets)

        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings.astype('float32'))
        
        self.wiki_title_metadata = all_titles_data # Store pageid, title, snippet
        
        faiss.write_index(self.faiss_index, WIKIPEDIA_TITLE_INDEX_PATH)
        np.save(WIKIPEDIA_TITLE_EMBEDDINGS_PATH, embeddings)
        with open(WIKIPEDIA_TITLE_METADATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.wiki_title_metadata, f, indent=4)
        print(f"Wikipedia title index built with {len(self.wiki_title_metadata)} entries.")

    def load_title_index(self):
        """Loads pre-built FAISS index for Wikipedia titles/summaries."""
        print("Loading Wikipedia title index...")
        try:
            self.faiss_index = faiss.read_index(WIKIPEDIA_TITLE_INDEX_PATH)
            self.wiki_title_metadata = json.load(open(WIKIPEDIA_TITLE_METADATA_PATH, 'r', encoding='utf-8'))
            print("Wikipedia title index loaded.")
            return True
    except FileNotFoundError:
        print(f"Wikipedia title index not found at {WIKIPEDIA_TITLE_INDEX_PATH} or {WIKIPEDIA_TITLE_METADATA_PATH}. Please build it first or check paths.")
        return False
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for Wikipedia title metadata from {WIKIPEDIA_TITLE_METADATA_PATH}: {e}")
            return False
        except Exception as e:
        print(f"An unexpected error occurred while loading Wikipedia title index: {e}")
            return False


    def _get_tfidf_keywords(self, text: str, top_n: int = 5) -> list[str]:
        """
        Extracts top_n keywords from text using TF-IDF.
        Uses preprocessing with lemmatization.
        """
        # Use original text for TF-IDF, preprocessing (including lemmatization) is done here
        preprocessed_text_for_tfidf = preprocess_text(text, perform_lemmatization=True)
        if not preprocessed_text_for_tfidf.strip(): # Check if empty after preprocessing
            return []

        try:
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.95, min_df=1)
            tfidf_matrix = vectorizer.fit_transform([preprocessed_text_for_tfidf])
            feature_names = vectorizer.get_feature_names_out()

            if feature_names.size == 0: # Check if feature_names is empty (e.g. text was only stopwords)
                return []

            scores = tfidf_matrix.toarray().flatten()

            sorted_indices = np.argsort(scores)[::-1]

            top_keywords = []
            # Iterate up to top_n or fewer if not enough terms meet criteria
            for i in sorted_indices:
                if len(top_keywords) >= top_n:
                    break
                # Add a score threshold if needed, e.g. scores[i] > 0.1
                if scores[i] > 0.0: # Basic check to ensure term has a score
                     top_keywords.append(feature_names[i])
            return top_keywords
        except ValueError as e: # Catches errors like empty vocabulary
            print(f"TF-IDF ValueError for input text snippet '{text[:50]}...': {e}")
            return []

    def retrieve_pages(self, query_text: str) -> list[dict]:
        """
        Retrieves relevant Wikipedia pages for a given query text.

        The process involves:
        1. Preprocessing the query text.
        2. Augmenting the query with Named Entities (NER) and TF-IDF keywords.
        3. Searching Wikipedia using its API with the augmented query.
        4. Embedding the search results (titles + snippets) and the original query's
           base form (`base_s_query`).
        5. Re-ranking the search results based on semantic similarity between the
           query embedding and result embeddings.
        6. Fetching the full text content of the top-ranked, unique pages.
        7. Caching downloaded page content.

        Args:
            query_text (str): The input question or query.

        Returns:
            list[dict]: A list of dictionaries, where each dictionary represents a
                        retrieved Wikipedia page and contains its 'id', 'title', and
                        'text'. Returns an empty list if no relevant pages are found
                        or if an error occurs during the process.
        """
        if not isinstance(query_text, str) or not query_text.strip():
            print("Warning: retrieve_pages received empty or invalid query_text. Returning empty list.")
            return []

        # Step 1: Formulate search query
        # Base query: less aggressive preprocessing (e.g., no lemmatization for main search query to Wikipedia)
        base_s_query = preprocess_text(query_text, perform_lemmatization=False)

        # NER Entities from original query_text
        entities_tuples = extract_medical_entities_spacy(query_text)
        # Get unique, lowercased entity texts from specified types
        ner_entities_texts = list(set([ent[0].lower() for ent in entities_tuples if ent[1] in RELEVANT_ENTITY_TYPES]))

        # TF-IDF Keywords from original query_text (preprocessing is internal to _get_tfidf_keywords)
        tfidf_keywords = self._get_tfidf_keywords(query_text, top_n=3) # top_n=3 for fewer, high-quality terms

        # Combine terms for the final search query for Wikipedia API
        # Start with base query terms (split, lowercased for set operations)
        final_query_terms = base_s_query.lower().split()
        existing_terms_set = set(final_query_terms) # Keep track of terms already added

        # Add NER entities (if not already present from base query)
        for term in ner_entities_texts: # ner_entities_texts are already lowercased
            if term not in existing_terms_set:
                final_query_terms.append(term)
                existing_terms_set.add(term)

        # Add TF-IDF keywords (if not already present)
        for term in tfidf_keywords: # tfidf_keywords can be mixed case
            term_lower = term.lower()
            if term_lower not in existing_terms_set:
                final_query_terms.append(term_lower)
                existing_terms_set.add(term_lower)

        search_query_for_api = " ".join(final_query_terms)
        
        # Logging the components of the query for transparency
        print(f"Original Query: '{query_text}'")
        print(f"Base Search Query (for API and embedding): '{base_s_query}'")
        print(f"NER Entities (lower, unique): {ner_entities_texts}")
        print(f"TF-IDF Keywords (raw, top_n=3): {tfidf_keywords}")
        print(f"Final Augmented Wikipedia Search Query (for API): '{search_query_for_api}'")

        # Step 2: Use Wikipedia API for initial search (online step)
        search_results = self._get_wikipedia_search_results(search_query_for_api)
        if not search_results:
            print("No search results from Wikipedia API for augmented query.")
            return []

        # Step 3: Embed search results (titles + snippets) and query, then rank
        # The query to embed for semantic ranking should be 'base_s_query' (preprocessed original query),
        # as it represents the core intent better than the keyword-stuffed 'search_query_for_api'.
        query_embedding_text = base_s_query

        result_texts = [f"{s['title']} {s['snippet']}" for s in search_results]

        if not result_texts: # Handle empty result_texts before attempting to get embeddings
            print("No text from search results to embed. Skipping ranking.")
            ranked_indices = [] # Ensure ranked_indices is defined
            result_embeddings = np.array([]) # Ensure result_embeddings is defined
            query_embedding = self.get_embeddings([query_embedding_text]) # Still embed query for consistency if needed later
        else:
            result_embeddings = self.get_embeddings(result_texts)
            query_embedding = self.get_embeddings([query_embedding_text])

        # If FAISS index is not built from a comprehensive set of titles,
        # we can do a direct similarity search against the current search results.
        # For a full FAISS index for titles, you'd perform a search on self.faiss_index.
        
        # Here, we'll re-rank the `search_results` using semantic similarity to the query.
        # Ensure query_embedding is 2D array for dot product.
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        # Ensure result_embeddings is also 2D, even if only one result.
        if result_embeddings.ndim == 1 and result_embeddings.size > 0 : # Check size to avoid issues with np.array([])
             result_embeddings = np.expand_dims(result_embeddings, axis=0)

        # Check if result_embeddings is empty or not 2D before dot product
        if not isinstance(result_embeddings, np.ndarray) or result_embeddings.ndim != 2 or result_embeddings.shape[0] == 0:
            print("No valid embeddings generated for search results. Skipping ranking.")
            # If no valid embeddings, we might take results as they came from API, or return empty.
            # For now, proceed with empty ranked_indices, meaning no pages selected based on semantic similarity.
            ranked_indices = []
        else:
            try:
                similarities = np.dot(query_embedding, result_embeddings.T).flatten()
                ranked_indices = np.argsort(similarities)[::-1] # Get indices of highest similarity first
            except ValueError as e:
                print(f"Error during similarity calculation (np.dot): {e}. Shapes: query_embedding={query_embedding.shape}, result_embeddings={result_embeddings.shape}")
                ranked_indices = []


        retrieved_articles = []
        seen_titles = set()

        for idx in ranked_indices:
            if len(retrieved_articles) >= MAX_WIKIPEDIA_PAGES_PER_QUESTION:
                break
            
            candidate_page = search_results[idx]
            title = candidate_page["title"]

            if title not in seen_titles:
                # Step 4: Download full content for the top 5 unique pages (online step per page)
                print(f"  Fetching full content for: '{title}'...")
                full_text = self._get_wikipedia_page_content(title)
                if full_text:
                    retrieved_articles.append({
                        "id": candidate_page.get("pageid", title.replace(" ", "_")),
                        "title": title,
                        "text": full_text
                    })
                    seen_titles.add(title)
                else:
                    print(f"  Warning: Could not fetch content for '{title}'.")
            
        print(f"Successfully retrieved and cached {len(retrieved_articles)} unique Wikipedia pages.")
        return retrieved_articles

if __name__ == "__main__":
    # --- Example: How to pre-download models ---
    print("Pre-downloading Sentence Transformer model...")
    os.makedirs(SENTENCE_TRANSFORMER_MODEL_PATH, exist_ok=True)
    tokenizer_st = AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER_MODEL)
    model_st = AutoModel.from_pretrained(SENTENCE_TRANSFORMER_MODEL)
    tokenizer_st.save_pretrained(SENTENCE_TRANSFORMER_MODEL_PATH)
    model_st.save_pretrained(SENTENCE_TRANSFORMER_MODEL_PATH)
    print(f"Sentence Transformer model saved to {SENTENCE_TRANSFORMER_MODEL_PATH}")

    from transformers import AutoModelForQuestionAnswering
    print("Pre-downloading QA model...")
    os.makedirs(QA_MODEL_PATH, exist_ok=True)
    tokenizer_qa = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    model_qa = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
    tokenizer_qa.save_pretrained(QA_MODEL_PATH)
    model_qa.save_pretrained(QA_MODEL_PATH)
    print(f"QA model saved to {QA_MODEL_PATH}")

    # --- Testing Wikipedia Retriever (with dynamic fetching) ---
    print("\n--- Testing Wikipedia Retriever with dynamic fetching ---")
    retriever = WikipediaRetriever()

    # (Optional) Build a conceptual title index if you want to pre-populate for FAISS search on titles
    # This requires an internet connection for initial search terms
    # retriever.build_title_index() # Run this once if you decide to have a pre-built title FAISS index

    # Perform a sample retrieval (requires internet connection for API calls)
    query = "What causes type 2 diabetes?"
    retrieved_pages = retriever.retrieve_pages(query)

    print(f"\nRetrieved pages for '{query}':")
    if retrieved_pages:
        for page in retrieved_pages:
            print(f"- {page['title']}")
            print(f"  Content snippet: {page['text'][:100]}...")
    else:
        print("No pages retrieved.")