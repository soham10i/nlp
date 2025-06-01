import argparse
from data_loader import load_medqa_dataset
from preprocessing import preprocess_text, extract_medical_entities_spacy
from retriever import WikipediaRetriever
from reader import QAModelReader
from answer_selector import AnswerSelector
from evaluator import Evaluator
from config import SUBSET_EVAL_SIZE, MAX_WIKIPEDIA_PAGES_PER_QUESTION, WIKIPEDIA_CACHE_DIR, \
                   SENTENCE_TRANSFORMER_MODEL, QA_MODEL_NAME, SPACY_MODEL_NAME, \
                   SENTENCE_TRANSFORMER_MODEL_PATH, QA_MODEL_PATH # Added for pre-downloading in main
import os
# import torch # torch is imported by specific modules if/when needed, not directly used in main.py's top level
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering # Added for pre-downloading
import spacy # For SpaCy model download check

def pre_download_all_models():
    """
    Ensures all necessary Hugging Face Transformer models and SpaCy language models
    are downloaded and cached locally.

    This function attempts to download:
    - Sentence Transformer model (for retriever and answer selector).
    - Question Answering (QA) model (for reader).
    - SpaCy language model (for preprocessing).

    Models are saved to paths defined in `config.py`.
    Errors during download are caught and reported, allowing the script to
    potentially continue if some models are already present or if manual
    download is preferred.
    """
    print("Ensuring all models are pre-downloaded...")
    
    # Sentence Transformer
    os.makedirs(SENTENCE_TRANSFORMER_MODEL_PATH, exist_ok=True)
    try:
        AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER_MODEL).save_pretrained(SENTENCE_TRANSFORMER_MODEL_PATH) 
        AutoModel.from_pretrained(SENTENCE_TRANSFORMER_MODEL).save_pretrained(SENTENCE_TRANSFORMER_MODEL_PATH) 
        print(f"Sentence Transformer model saved to {SENTENCE_TRANSFORMER_MODEL_PATH}")
    except Exception as e:
        print(f"Error downloading Sentence Transformer model: {e}")

    # QA Model
    os.makedirs(QA_MODEL_PATH, exist_ok=True)
    try:
        AutoTokenizer.from_pretrained(QA_MODEL_NAME).save_pretrained(QA_MODEL_PATH)  
        AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME).save_pretrained(QA_MODEL_PATH)  
        print(f"QA model saved to {QA_MODEL_PATH}")
    except Exception as e:
        print(f"Error downloading QA model: {e}")

    # SpaCy Model (handled in preprocessing.py for now, but can be added here)
    try:
        import spacy
        spacy.load(SPACY_MODEL_NAME)
        print(f"SpaCy model '{SPACY_MODEL_NAME}' is available.")
    except OSError:
        print(f"SpaCy model '{SPACY_MODEL_NAME}' not found locally. Attempting download...")
        try:
            spacy.cli.download(SPACY_MODEL_NAME)
            print(f"SpaCy model '{SPACY_MODEL_NAME}' downloaded.")
        except Exception as e:
            print(f"Error downloading SpaCy model: {e}. Please try 'python -m spacy download {SPACY_MODEL_NAME}' manually.")


def main():
    """
    Main function to run the Medical Question Answering System.

    Orchestrates the entire pipeline:
    1. Parses command-line arguments for evaluation mode (full/subset) and
       optional Wikipedia title index building.
    2. Ensures necessary directories and models are set up/downloaded.
    3. Initializes the Retriever, Reader, AnswerSelector, and Evaluator components.
    4. Loads the MedQA dataset.
    5. Iterates through questions, performing retrieval, reading, and answer selection.
    6. Handles errors gracefully at each stage for each question, allowing the
       pipeline to continue with the next question.
    7. Evaluates the system's performance (accuracy) and prints the results.

    Command-line arguments:
        --subset_eval: Evaluate on a smaller subset of questions for quick testing.
        --build_title_index: Build a FAISS index of Wikipedia titles (optional,
                             requires internet for initial data fetching).
    """
    parser = argparse.ArgumentParser(description="Medical Question Answering System")
    parser.add_argument("--subset_eval", action="store_true",
                        help=f"Evaluate on a subset of {SUBSET_EVAL_SIZE} questions instead of all.") 
    parser.add_argument("--build_title_index", action="store_true",
                        help="Build Wikipedia title index (run once to create FAISS index of titles/snippets).")
    args = parser.parse_args()

    # Ensure directories exist
    os.makedirs("models/sentence_transformers", exist_ok=True)
    os.makedirs("models/qa_models", exist_ok=True)
    os.makedirs("models/spacy", exist_ok=True)
    os.makedirs(WIKIPEDIA_CACHE_DIR, exist_ok=True)
    os.makedirs("index", exist_ok=True)

    # Pre-download all necessary models (requires internet)
    pre_download_all_models()

    print("Initializing components...")
    retriever = WikipediaRetriever()
    
    # Build title index if requested (conceptual, assumes you might pre-populate with medical titles)
    if args.build_title_index:
        print("Building Wikipedia title index (requires internet for initial searches)...")
        retriever.build_title_index()
    else:
        # Attempt to load pre-built title index, if it's meant to be persistent
        retriever.load_title_index()


    reader = QAModelReader()
    selector = AnswerSelector()
    evaluator = Evaluator()

    # Load MedQA test dataset
    medqa_test_data = load_medqa_dataset(split="test") 

    if args.subset_eval:
        print(f"\nEvaluating on a subset of {SUBSET_EVAL_SIZE} questions...")
        medqa_test_data = medqa_test_data.select(range(SUBSET_EVAL_SIZE)) 
    else:
        print("\nEvaluating on the full test dataset...")

    predictions = []
    ground_truth_labels = []

    for i, question_entry in enumerate(medqa_test_data):
        print(f"\n--- Processing Question {i+1}/{len(medqa_test_data)} (ID: {question_entry['id']}) ---")
        question_text = question_entry['sent1']
        options = [question_entry[f'ending{j}'] for j in range(4)]
        correct_label = question_entry['label']

        print(f"Question: {question_text}")
        print(f"Options: {options}")
        print(f"Correct Answer Index: {correct_label}")

        # 1. Retrieval (requires internet connection for Wikipedia API calls)
        retrieved_pages = []
        try:
            retrieved_pages = retriever.retrieve_pages(question_text) 
            if not retrieved_pages:
                print("No relevant Wikipedia pages retrieved. Skipping question.")
                predictions.append(0) # Default or random choice if no pages
                ground_truth_labels.append(correct_label)
                continue
        except Exception as e:
            print(f"Error during retrieval for question {question_entry['id']}: {e}. Skipping question.")
            predictions.append(0)
            ground_truth_labels.append(correct_label)
            continue

        # 2. Reading/Answer Extraction
        candidate_answers_from_reader = []
        try:
            candidate_answers_from_reader = reader.extract_answers_from_pages(question_text, retrieved_pages) 
            if not candidate_answers_from_reader:
                print("No answer spans extracted by reader. Skipping question.")
                predictions.append(0) # Default or random choice if no answers
                ground_truth_labels.append(correct_label)
                continue
        except Exception as e:
            print(f"Error during answer extraction for question {question_entry['id']}: {e}. Skipping question.")
            predictions.append(0)
            ground_truth_labels.append(correct_label)
            continue

        # 3. Answer Selection
        selected_option_index = 0
        try:
            selected_option_index = selector.select_best_answer(question_text, options, candidate_answers_from_reader)
        except Exception as e:
            print(f"Error during answer selection for question {question_entry['id']}: {e}. Defaulting to option 0.")
            selected_option_index = 0 # Default if selection fails
            
        predictions.append(selected_option_index)
        ground_truth_labels.append(correct_label)

    # 4. Evaluation
    accuracy = evaluator.evaluate(predictions, ground_truth_labels) 

    print("\n--- Project Work Complete ---")
    print(f"Final Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # NLTK and SpaCy data downloads are handled in preprocessing.py for clarity.
    # The initial model downloads are handled by pre_download_all_models() in main.py
    # This main script assumes internet access for Wikipedia API calls during runtime.
    main()