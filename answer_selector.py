from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from config import SENTENCE_TRANSFORMER_MODEL, SENTENCE_TRANSFORMER_MODEL_PATH, DEVICE

class AnswerSelector:
    def __init__(self):
        """
        Initializes the AnswerSelector.

        Loads the Sentence Transformer model used for calculating semantic similarity
        between candidate answers and multiple-choice options.
        Errors during model download/loading from Hugging Face Hub may cause
        program termination.
        """
        # Load Sentence Transformer for semantic similarity
        print(f"Loading Sentence Transformer for AnswerSelector to {DEVICE}...")
        self.tokenizer_st = AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER_MODEL_PATH)
        self.model_st = AutoModel.from_pretrained(SENTENCE_TRANSFORMER_MODEL_PATH).to(DEVICE) 
        self.model_st.eval()
        # Determine embedding dimension for handling empty text cases
        try:
            # Encode a dummy text to find the embedding dimension
            dummy_embedding = self.get_embedding_st("test")
            self.embedding_dim = dummy_embedding.shape[0]
        except Exception as e:
            print(f"Warning: Could not determine embedding dimension during AnswerSelector init: {e}")
            # Fallback to a common dimension for MiniLM or allow it to fail later if ST model truly failed
            self.embedding_dim = 384 # Common for all-MiniLM-L6-v2

    def _mean_pooling(self, model_output: tuple, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs mean pooling on token embeddings to get sentence embeddings.
        (Helper function for Sentence Transformer).

        Args:
            model_output (tuple): Output from the Hugging Face model.
            attention_mask (torch.Tensor): Attention mask for the input tokens.

        Returns:
            torch.Tensor: Sentence embeddings.
        """
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_embedding_st(self, text: str) -> np.ndarray:
        """
        Generates an embedding for a single text string using the Sentence Transformer model.

        Args:
            text (str): The input text.

        Returns:
            np.ndarray: A 1D NumPy array representing the sentence embedding.
                        Returns a zero vector of `self.embedding_dim` if the text is empty,
                        invalid, or if embedding generation fails.
        """
        if not isinstance(text, str) or not text.strip():
            # print(f"Warning: get_embedding_st received empty or invalid text. Returning zero vector.")
            return np.zeros(self.embedding_dim, dtype=np.float32)
        try:
            encoded_input = self.tokenizer_st(text, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
            with torch.no_grad():
                model_output = self.model_st(**encoded_input)
            sentence_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
            return sentence_embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error generating sentence embedding for text '{text[:50]}...': {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)


    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates cosine similarity between the Sentence Transformer embeddings of two texts.

        Args:
            text1 (str): The first text string.
            text2 (str): The second text string.

        Returns:
            float: The cosine similarity score (between -1.0 and 1.0).
                   Returns 0.0 if embeddings cannot be generated or are zero vectors.
        """
        embedding1 = self.get_embedding_st(text1)
        embedding2 = self.get_embedding_st(text2)

        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            # This handles cases where one or both texts resulted in a zero embedding (e.g., empty text)
            return 0.0

        # Normalize embeddings to unit vectors for cosine similarity
        embedding1_normalized = embedding1 / norm1
        embedding2_normalized = embedding2 / norm2

        similarity = np.dot(embedding1_normalized, embedding2_normalized)
        return float(similarity) # Ensure float type

    def select_best_answer(self, question: str, multiple_choice_options: list[str], candidate_answers: list[dict]) -> int:
        """
        Selects the best answer option from multiple-choice based on extracted candidates. [cite: 237]

        Args:
            question (str): The original medical question.
            multiple_choice_options (list): List of 4 answer strings. [cite: 126, 240]
            candidate_answers (list): List of dicts from reader, e.g.,
                                     [{"answer": "...", "score": ..., "source_page_title": "..."}, ...]

        Returns:
            int: The index (0-3) of the selected best answer option.
        """
        # Basic input validation
        if not isinstance(multiple_choice_options, list) or not multiple_choice_options:
            print("Warning: select_best_answer received empty or invalid multiple_choice_options. Defaulting to option index 0.")
            return 0
        if not all(isinstance(opt, str) for opt in multiple_choice_options):
            print("Warning: select_best_answer received multiple_choice_options with non-string elements. Proceeding cautiously.")
            # Convert to string to be safe, or could error out
            multiple_choice_options = [str(opt) for opt in multiple_choice_options]


        if not isinstance(candidate_answers, list):
            print("Warning: select_best_answer received non-list candidate_answers. Defaulting to option index 0.")
            return 0

        option_scores = [0.0] * len(multiple_choice_options)
        
        # Consider each extracted candidate answer
        for cand_ans in candidate_answers:
            if not isinstance(cand_ans, dict) or 'answer' not in cand_ans or 'score' not in cand_ans:
                print(f"Warning: Skipping invalid candidate answer data: {cand_ans}")
                continue

            extracted_text = cand_ans.get('answer', "")
            qa_score = cand_ans.get('score', 0.0)

            if not isinstance(extracted_text, str) or not extracted_text.strip():
                continue # Skip candidate answers with no text

            if not isinstance(qa_score, (float, int)):
                try:
                    qa_score = float(qa_score)
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert qa_score '{qa_score}' to float. Using 0.0.")
                    qa_score = 0.0


            for i, option_text_orig in enumerate(multiple_choice_options):
                option_text = str(option_text_orig) # Ensure option_text is a string for processing

                # 1. Exact Matching (highest priority)
                if extracted_text.lower() == option_text.lower():
                    option_scores[i] += qa_score * 2.0 # Boost exact matches
                    # print(f"  Exact match found: '{extracted_text}' for option '{option_text}'")
                    continue # Process this candidate against other options if needed, or assume one best match per candidate?
                             # Current logic: one candidate can contribute to multiple options if it matches multiple.

                # 2. Fuzzy Matching
                fuzzy_ratio = fuzz.ratio(extracted_text.lower(), option_text.lower()) / 100.0
                if fuzzy_ratio > 0.8: # Threshold for fuzzy match
                    option_scores[i] += qa_score * (1.0 + fuzzy_ratio) # Boost based on fuzzy match quality
                    # print(f"  Fuzzy match found: '{extracted_text}' with '{option_text}' (Ratio: {fuzzy_ratio:.2f})")
                    continue

                # 3. Semantic Similarity (if fuzzy/exact not strong enough)
                # This is called even if fuzzy match was found above 0.8; consider if this is desired.
                # For now, if an exact or strong fuzzy match occurs, we continue, so semantic sim is only for weaker/no fuzzy.
                # The prompt implies to use this if "fuzzy/exact not strong enough".
                # The current `continue` statements mean semantic similarity is only computed if exact/fuzzy criteria are not met.
                semantic_sim = self.calculate_semantic_similarity(extracted_text, option_text)
                if semantic_sim > 0.5: # Threshold for semantic similarity
                    option_scores[i] += qa_score * (0.5 + semantic_sim) # Smaller boost than exact/fuzzy
                    # print(f"  Semantic match found: '{extracted_text}' with '{option_text}' (Sim: {semantic_sim:.2f})")
        
        # Find the index of the option with the highest aggregated score
        if not option_scores or max(option_scores) <= 0: # Changed to <= 0 to handle cases where all scores are zero or negative (though unlikely with current logic)
            print(f"No strong match found for question '{question[:50]}...'. Option scores: {option_scores}. Defaulting to first option (index 0).")
            return 0 # Fallback if no options score highly or no candidates
        
        best_option_index = int(np.argmax(option_scores)) # Ensure int
        print(f"Option scores: {option_scores}")
        print(f"Selected option index: {best_option_index} ('{multiple_choice_options[best_option_index]}')")
        return best_option_index

if __name__ == "__main__":
    # Ensure ST model is downloaded for this to run independently
    selector = AnswerSelector()

    options = ["metabolism of carbohydrates", "regulation of blood pressure", "bone density", "muscle contraction"]
    
    # Candidate answers from the reader
    candidate_answers = [
        {"answer": "regulates the metabolism of carbohydrates", "score": 0.95, "source_page_title": "Insulin"},
        {"answer": "main anabolic hormone of the body", "score": 0.80, "source_page_title": "Insulin"},
        {"answer": "located in the abdomen behind the stomach", "score": 0.10, "source_page_title": "Pancreas"} # Low confidence/irrelevant
    ]
    question = "What is the main function of insulin?"

    print(f"Question: {question}")
    print(f"Options: {options}")
    print(f"Candidate Answers: {candidate_answers}")

    selected_index = selector.select_best_answer(question, options, candidate_answers)
    print(f"\nFinal Selected Answer: {options[selected_index]}")

    options_2 = ["cardiac arrest", "myocardial infarction", "stroke", "appendicitis"]
    candidate_answers_2 = [
        {"answer": "myocardial infarction", "score": 0.99, "source_page_title": "Heart attack"},
        {"answer": "decreased blood flow to heart", "score": 0.85, "source_page_title": "Heart attack"},
    ]
    question_2 = "What is another name for a heart attack?"

    print(f"\n--- Second Example ---")
    print(f"Question: {question_2}")
    print(f"Options: {options_2}")
    print(f"Candidate Answers: {candidate_answers_2}")

    selected_index_2 = selector.select_best_answer(question_2, options_2, candidate_answers_2)
    print(f"\nFinal Selected Answer: {options_2[selected_index_2]}")