# Medical Question Answering System with Live Wikipedia Fetching

## 1. Overview

This project implements a question answering system specifically designed for the medical domain, using the MedQA (USMLE-style) dataset. It leverages live Wikipedia fetching to gather relevant information dynamically for each question. The system preprocesses questions, retrieves articles from Wikipedia, uses an extractive QA model to find potential answers in these articles, and then selects the best multiple-choice option.

## 2. Project Structure

```
/project_root
  main.py             # Main orchestration script
  data_loader.py      # Loads MedQA dataset
  preprocessing.py    # Text preprocessing utilities (clean, tokenize, NER)
  retriever.py        # Fetches relevant Wikipedia articles
  reader.py           # Extracts answer spans from articles using a QA model
  answer_selector.py  # Selects the best multiple-choice option
  evaluator.py        # Evaluates the system's accuracy
  config.py           # Configuration settings (model names, paths, etc.)
  requirements.txt    # Python dependencies
  README.md           # This file
  /models/            # Stores downloaded ML models
  /wikipedia_cache/   # Caches downloaded Wikipedia articles
  /index/             # Stores FAISS index files (if pre-built title index is used)
```

## 3. Setup

### 3.1. Prerequisites
*   Python 3.8+
*   Access to the internet (for downloading models, datasets, and fetching Wikipedia articles live)

### 3.2. Installation
1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd <repository_directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install all necessary Python packages, including PyTorch, Transformers, spaCy, NLTK, Scikit-learn, Wikipedia, etc.

### 3.3. Model and Data Download
The system requires several machine learning models (Sentence Transformer, QA model, spaCy language model) and NLTK data.

*   **Automated Download via `main.py`:**
    The first time you run `main.py`, it will attempt to download and save all required models to the `./models/` directory and NLTK data if not found.
    ```bash
    python main.py --subset_eval # Using --subset_eval for a quicker first run
    ```
    Ensure you have an active internet connection.

*   **Manual SpaCy Model Download (if needed):**
    If the automated spaCy model download fails, you can try it manually:
    ```bash
    python -m spacy download en_core_web_sm
    ```
    (The `config.py` specifies `en_core_web_sm`. If you change it, download the corresponding model).

## 4. Running the System

The main script for running the QA pipeline is `main.py`.

### 4.1. Full Evaluation
To run the system on the full MedQA test dataset:
```bash
python main.py
```
This will process all questions, fetch Wikipedia articles live, and evaluate accuracy. This can take a significant amount of time and requires a stable internet connection.

### 4.2. Subset Evaluation
For a quicker test or debugging, run on a subset of questions (default is 1000, defined in `config.py`):
```bash
python main.py --subset_eval
```

### 4.3. (Optional) Building Wikipedia Title Index
The retriever has functionality to build a FAISS index of Wikipedia titles/snippets if you want to experiment with a pre-defined set of relevant pages. This is not the primary mode of operation for live fetching but is available.
```bash
python main.py --build_title_index
```
This requires internet access to fetch initial search results for index creation.

## 5. Modules

*   **`main.py`**: Orchestrates the entire QA pipeline: loads data, initializes components, processes questions, and triggers evaluation. Handles model downloads.
*   **`data_loader.py`**: Responsible for loading the MedQA dataset from Hugging Face's `datasets` library.
*   **`preprocessing.py`**: Contains functions for text cleaning (lowercasing, punctuation removal), tokenization, stopword removal, lemmatization, and Named Entity Recognition (NER) using spaCy.
*   **`retriever.py`**: Given a question, it preprocesses the query, augments it with NER entities and TF-IDF keywords, searches Wikipedia via its API, ranks search results using semantic similarity, and fetches the full text of the top N articles. Caches downloaded articles.
*   **`reader.py`**: Takes the retrieved Wikipedia articles and the question. It uses an extractive Question Answering (QA) model (e.g., DistilBERT-SQuAD) to identify potential answer spans within the text of each article. Handles chunking of long articles.
*   **`answer_selector.py`**: Aggregates all candidate answer spans from the `Reader`. It then compares each candidate to the 4 multiple-choice options provided by the MedQA question, using exact match, fuzzy string matching, and semantic similarity. It selects the MedQA option with the highest aggregated score.
*   **`evaluator.py`**: Calculates the system's performance, primarily accuracy (correct answer selection rate).
*   **`config.py`**: Centralized configuration for model names, file paths, API parameters, and other settings.

## 6. Expected Output

When you run `main.py`, you will see logging output for each question being processed, including:
*   The question and options.
*   The constructed Wikipedia search query.
*   Titles of retrieved Wikipedia pages.
*   Candidate answers extracted by the reader.
*   The final selected answer option.

At the end of the run, evaluation results will be printed:
```
--- Evaluation Results ---
Total Questions: [Number of questions processed]
Correctly Answered: [Number of correct answers]
Accuracy: [Accuracy score, e.g., 0.XXXX]

--- Project Work Complete ---
Final Accuracy: [Accuracy score, e.g., 0.XXXX]
```

## 7. Notes
*   The system relies on live calls to the Wikipedia API. Ensure your network allows this. Rate limiting or IP blocks from Wikipedia are a potential risk if making too many requests too quickly (though the `wikipedia` library and `requests` sessions usually handle this gracefully for moderate use).
*   Performance (accuracy) will depend on the quality of retrieved articles, the QA model's capabilities, and the effectiveness of the answer selection logic.
```
