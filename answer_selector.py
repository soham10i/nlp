from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from config import SENTENCE_TRANSFORMER_MODEL, SENTENCE_TRANSFORMER_MODEL_PATH, DEVICE

class AnswerSelector:
    def __init__(self):
        # Load Sentence Transformer for semantic similarity
        print(f"Loading Sentence Transformer for AnswerSelector to {DEVICE}...")
        self.tokenizer_st = AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER_MODEL_PATH)
        self.model_st = AutoModel.from_pretrained(SENTENCE_TRANSFORMER_MODEL_PATH).to(DEVICE) 
        self.model_st.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embedding_st(self, text):
        """Generates embedding for a single text using the Sentence Transformer model."""
        encoded_input = self.tokenizer_st(text, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            model_output = self.model_st(**encoded_input)
        sentence_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embedding.cpu().numpy().flatten() 

    def calculate_semantic_similarity(self, text1, text2):
        """Calculates cosine similarity between two texts' embeddings. [cite: 243, 244]"""
        embedding1 = self.get_embedding_st(text1)
        embedding2 = self.get_embedding_st(text2)
        # Normalize embeddings to unit vectors for cosine similarity
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        return np.dot(embedding1, embedding2)

    def select_best_answer(self, question, multiple_choice_options, candidate_answers):
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
        option_scores = [0.0] * len(multiple_choice_options)
        
        # Consider each extracted candidate answer
        for cand_ans in candidate_answers:
            extracted_text = cand_ans['answer']
            qa_score = cand_ans['score'] # Confidence score from the QA model [cite: 245]

            for i, option_text in enumerate(multiple_choice_options):
                # 1. Exact Matching (highest priority) [cite: 241]
                if extracted_text.lower() == option_text.lower():
                    option_scores[i] += qa_score * 2.0 # Boost exact matches
                    print(f"  Exact match found: '{extracted_text}' for option '{option_text}'")
                    continue # Move to next option for this candidate answer

                # 2. Fuzzy Matching [cite: 242]
                fuzzy_ratio = fuzz.ratio(extracted_text.lower(), option_text.lower()) / 100.0
                if fuzzy_ratio > 0.8: # Threshold for fuzzy match
                    option_scores[i] += qa_score * (1.0 + fuzzy_ratio) # Boost based on fuzzy match quality
                    print(f"  Fuzzy match found: '{extracted_text}' with '{option_text}' (Ratio: {fuzzy_ratio:.2f})")
                    continue

                # 3. Semantic Similarity (if fuzzy/exact not strong enough) [cite: 243, 244]
                semantic_sim = self.calculate_semantic_similarity(extracted_text, option_text)
                if semantic_sim > 0.5: # Threshold for semantic similarity
                    option_scores[i] += qa_score * (0.5 + semantic_sim) # Smaller boost than exact/fuzzy
                    print(f"  Semantic match found: '{extracted_text}' with '{option_text}' (Sim: {semantic_sim:.2f})")
        
        # Find the index of the option with the highest aggregated score [cite: 247]
        if not option_scores or max(option_scores) == 0:
            print("No strong match found. Defaulting to first option (index 0).")
            return 0 # Fallback if no options score highly or no candidates
        
        best_option_index = np.argmax(option_scores)
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