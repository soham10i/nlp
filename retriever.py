import os
import json
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from config import SENTENCE_TRANSFORMER_MODEL, SENTENCE_TRANSFORMER_MODEL_PATH, \
                   WIKIPEDIA_TITLE_EMBEDDINGS_PATH, WIKIPEDIA_TITLE_INDEX_PATH, \
                   WIKIPEDIA_TITLE_METADATA_PATH, DEVICE, WIKIPEDIA_SEARCH_RESULTS_TO_CONSIDER, \
                   MAX_WIKIPEDIA_PAGES_PER_QUESTION, WIKIPEDIA_CACHE_DIR, QA_MODEL_NAME, QA_MODEL_PATH
from preprocessing import preprocess_text, extract_medical_entities_spacy

class WikipediaRetriever:
    def __init__(self):
        print(f"Loading Sentence Transformer model to {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER_MODEL_PATH)
        self.model = AutoModel.from_pretrained(SENTENCE_TRANSFORMER_MODEL_PATH).to(DEVICE)
        self.model.eval() # Set model to evaluation mode

        self.faiss_index = None
        self.wiki_title_metadata = [] # List of {"title": ..., "pageid": ..., "summary": ...}

        # Ensure cache directory exists
        os.makedirs(WIKIPEDIA_CACHE_DIR, exist_ok=True)

    def _mean_pooling(self, model_output, attention_mask):
        """Performs mean pooling to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts):
        """Generates embeddings for a list of texts using the Sentence Transformer model."""
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings.cpu().numpy()

    def _get_wikipedia_search_results(self, query):
        """
        Performs an online search using the Wikipedia API for page titles/summaries. [cite: 55]
        Returns a list of dictionaries with 'title', 'pageid', 'snippet'.
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
            print(f"Error during Wikipedia API search for '{query}': {e}")
            return []

    def _get_wikipedia_page_content(self, title):
        """
        Downloads the full plain text content of a Wikipedia page. [cite: 55]
        Caches content locally.
        """
        # Check cache first
        cached_filepath = os.path.join(WIKIPEDIA_CACHE_DIR, f"{title.replace(' ', '_')}.txt")
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
            print(f"Error during Wikipedia API content fetch for '{title}': {e}")
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
        except FileNotFoundError as e:
            print(f"Wikipedia title index not found: {e}. Please build it first.")
            return False
        except Exception as e:
            print(f"Error loading Wikipedia title index: {e}")
            return False

    def retrieve_pages(self, query_text):
        """
        Performs semantic search for Wikipedia titles, downloads top 5 pages.
        """
        # Step 1: Formulate search query using preprocessed text and extracted entities
        # A simple approach: use the preprocessed question text as the query
        search_query = preprocess_text(query_text)
        entities = extract_medical_entities_spacy(query_text)
        if entities:
            # Augment query with key entities if relevant
            search_query += " " + " ".join([ent[0] for ent in entities if ent[1] in ["DISEASE", "DRUG", "ANATOMY"]])
        
        print(f"Searching Wikipedia for: '{search_query}'")

        # Step 2: Use Wikipedia API for initial search (online step)
        search_results = self._get_wikipedia_search_results(search_query)
        if not search_results:
            print("No search results from Wikipedia API.")
            return []

        # Step 3: Embed search results (titles + snippets) and query, then rank using FAISS
        result_texts = [f"{s['title']} {s['snippet']}" for s in search_results]
        result_embeddings = self.get_embeddings(result_texts)
        query_embedding = self.get_embeddings([search_query])

        # If FAISS index is not built from a comprehensive set of titles,
        # we can do a direct similarity search against the current search results.
        # For a full FAISS index for titles, you'd perform a search on self.faiss_index.
        
        # Here, we'll re-rank the `search_results` using semantic similarity to the query
        similarities = np.dot(query_embedding, result_embeddings.T).flatten()
        ranked_indices = np.argsort(similarities)[::-1] # Get indices of highest similarity first

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