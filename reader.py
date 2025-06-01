from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
from config import QA_MODEL_NAME, QA_MODEL_PATH, DEVICE

class QAModelReader:
    def __init__(self):
        """
        Initializes the QAModelReader.

        Loads the Question Answering model and tokenizer from Hugging Face based
        on configurations in `config.py`. It also sets up the QA pipeline
        and determines model-specific parameters like max sequence length and stride
        for context chunking.

        The constructor will print loading status and relies on Hugging Face's
        caching mechanism for model storage. If models are not found locally,
        they will be downloaded. Errors during model download/loading from
        Hugging Face Hub may cause program termination.
        """
        print(f"Loading QA model: {QA_MODEL_NAME} to {DEVICE}...") 
        self.tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_PATH)
        self.model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_PATH).to(DEVICE) 
        self.model.eval() # Set model to evaluation mode

        # Using Hugging Face pipeline for ease of use.
        # It automatically handles moving inputs to the specified device.
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if DEVICE == "cuda" else -1 # 0 for first GPU, -1 for CPU
        )
        # It's good practice to define model_max_length and stride either from config or as class attributes if they are fixed.
        # For now, let's use a common value for bert-based models or derive it.
        self.model_max_length = self.tokenizer.model_max_length
        # Define a reasonable stride, e.g., 1/4th of max length, or a fixed number like 128
        # This stride is for the context tokens, not the combined (question+context) sequence.
        self.doc_stride = self.model_max_length // 4
        print("QA model loaded.")

    def extract_answer(self, question: str, context: str) -> tuple[str | None, float]:
        """
        Extracts an answer span from the given context for the question.

        Handles long contexts by tokenizing the context and splitting it into
        overlapping chunks if its length (plus question length) exceeds the
        model's maximum sequence length. Each chunk is processed by the QA pipeline.
        The answer with the highest confidence score across all chunks is returned.

        Args:
            question (str): The question string.
            context (str): The context string from which to extract the answer.

        Returns:
            tuple[str | None, float]: A tuple containing the extracted answer text
                                      (or None if no answer is found) and its
                                      confidence score (0.0 if no answer).
        """
        if not isinstance(question, str) or not isinstance(context, str):
            print("Warning: extract_answer received non-string input for question or context.")
            return None, 0.0
        if not question.strip() or not context.strip():
            print("Warning: extract_answer received empty question or context.")
            return None, 0.0

        try:
            # Tokenize the question to find its length (in tokens)
            # We use encode to get IDs, not for direct input to pipeline here, but for length calculation.
            # The tokenizer for the pipeline will add special tokens ([CLS], [SEP])
            # For BERT: [CLS] question [SEP] context [SEP]
            # Max tokens for context = model_max_length - (tokens in question) - 3 special tokens
            question_input_ids = self.tokenizer.encode(question, add_special_tokens=False)

            # Calculate available length for context in each chunk
            # -3 for [CLS], [SEP] question [SEP] context [SEP]
            # More accurately, the tokenizer when preparing for the model will handle special tokens.
            # We should aim for context chunks that, when combined with the question by the pipeline's internal tokenizer,
            # fit within self.model_max_length.
            # A simpler approach for chunking text:
            # Estimate max context tokens, then decode token chunks to text.

            # Max length for the context part of the input
            # The pipeline's internal tokenizer will add special tokens.
            # For BERT: [CLS] Q_tokens [SEP] C_tokens [SEP].
            # So, effective_max_len_for_context_tokens = self.model_max_length - len(question_input_ids) - 3 (approx for CLS, SEP, SEP)
            # However, it's safer to let the pipeline tokenizer handle exact truncation or use its logic.
            # The Hugging Face pipeline itself doesn't expose a simple way to process pre-split token IDs for context chunks
            # while providing the question string separately. It expects context as a string.
            # So, we chunk the context string.

            context_tokens_ids = self.tokenizer.encode(context, add_special_tokens=False) # Get context token IDs without special tokens
            num_context_tokens = len(context_tokens_ids)

            # Estimate the max number of context tokens we can have in a single pass with the question
            # This is a rough estimation to decide if chunking is needed.
            # The pipeline will internally tokenize `question + context`.
            # A single tokenized sequence for QA pipeline: [CLS] q_toks [SEP] c_toks [SEP]
            # Max context tokens for one pass = model_max_length - question_tok_len - 3 (CLS, SEP, SEP)
            # This is a simplified view. The tokenizer might have more complex logic for allocating space.
            # Let's define max_chunk_context_tokens based on model_max_length, considering question length.
            # The tokenizer used by the pipeline will truncate the context if (question + context) is too long.
            # We want to avoid that truncation by feeding manageable context chunks.

            # Define a max length for the text chunks of context we create.
            # This is not the model_max_length directly, but a target for our context string chunks.
            # The tokenizer inside the pipeline will later combine this chunk with the question.
            # A practical approach: make context chunks somewhat smaller than model_max_length
            # to ensure they fit with the question.
            # Let's set chunk_target_token_length for context part.
            # We use a chunk_token_length for context tokens, then decode to string.
            # This should be less than model_max_length minus question length and special tokens.
            # For simplicity, let's make context chunks of size (model_max_length - len(question_input_ids) - safety_margin)
            # Safety margin for special tokens ([CLS], [SEP], [SEP]) and potential miscounts. Let's say 50 tokens.
            # This logic is getting complicated. A fixed max_chunk_text_length might be more robust initially.
            # Or, use the tokenizer's ability to work with strides if possible with string inputs.

            # Simpler approach: If the full context (when tokenized with question) would be truncated, then chunk.
            # The pipeline's tokenizer will truncate context if question + context > model_max_length.
            # Let's use the example's manual token splitting idea for context.

            # Max tokens the model can take for the context part, considering the question.
            # This is the length of context tokens that can fit with the question and special tokens.
            max_context_tokens_for_chunk = self.model_max_length - len(question_input_ids) - 3 # For [CLS], [SEP], [SEP]

            if max_context_tokens_for_chunk <= 0: # Question itself is too long or near max_length
                # Handle this edge case: maybe truncate question or return error.
                # For now, try to proceed with a very small context allowance if possible,
                # or rely on pipeline to error out if question itself is too long.
                # A more robust way would be to check question length against model_max_length first.
                print(f"Warning: Question might be too long for the model. Tokens: {len(question_input_ids)}")
                # If question itself (plus special tokens) already exceeds max_seq_length,
                # then no room for context. The pipeline will likely handle this by truncating the question or erroring.
                # Let's assume the question fits, and max_context_tokens_for_chunk is the budget for context tokens.
                # If max_context_tokens_for_chunk is very small (e.g. < stride), chunking might not be effective.
                if max_context_tokens_for_chunk < self.doc_stride // 2 and num_context_tokens > max_context_tokens_for_chunk : # If context exists but allowance is too small
                     print(f"Warning: Very small token allowance for context ({max_context_tokens_for_chunk}). May lead to poor chunking.")


            if num_context_tokens > max_context_tokens_for_chunk:
                all_chunk_results = []

                # Ensure stride is less than the chunk capacity for context
                current_stride = min(self.doc_stride, max_context_tokens_for_chunk - 1 if max_context_tokens_for_chunk > 0 else 0)
                if current_stride <= 0 and num_context_tokens > 0 : # If context exists but stride is non-positive (e.g. question too long)
                    current_stride = max_context_tokens_for_chunk // 2 # Fallback stride if possible
                    if current_stride <=0 : # Still bad, means max_context_tokens_for_chunk is ~0 or 1.
                         print(f"Error: Cannot process context due to extremely long question or model constraints. Context allowance: {max_context_tokens_for_chunk}")
                         return None, 0.0


                for i in range(0, num_context_tokens, max_context_tokens_for_chunk - current_stride):
                    start_token = i
                    end_token = min(i + max_context_tokens_for_chunk, num_context_tokens)

                    chunk_token_ids = context_tokens_ids[start_token:end_token]
                    # Decode back to string for the pipeline
                    chunk_text = self.tokenizer.decode(chunk_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    if not chunk_text.strip(): # Avoid processing empty or whitespace-only chunks
                        continue

                    # Run pipeline on this chunk
                    # The pipeline will tokenize `question` and `chunk_text` together here.
                    # And it will handle truncation if, for some reason, this specific chunk + question is still too long.
                    result = self.qa_pipeline(question=question, context=chunk_text)
                    if result and result['answer']: # Ensure answer is not None or empty
                        all_chunk_results.append(result)

                if not all_chunk_results:
                    return None, 0.0

                # Select the best answer based on score
                best_result = max(all_chunk_results, key=lambda x: x['score'])
                return best_result['answer'], best_result['score']
            else:
                # Context is short enough, process as before
                result = self.qa_pipeline(question=question, context=context)
                return result['answer'], result['score']

        except Exception as e:
            # It's useful to log the question and a snippet of context for debugging
            context_snippet = context[:100] + "..." if len(context) > 100 else context
            question_snippet = question[:100] + "..." if len(question) > 100 else question
            print(f"Error during QA extraction for question '{question_snippet}' and context snippet '{context_snippet}': {e}")
            # Optionally, re-raise or handle more gracefully depending on application needs
            # For robustness, especially with chunking, ensure we always return a tuple
            return None, 0.0

    def extract_answers_from_pages(self, question: str, wikipedia_pages: list[dict]) -> list[dict]:
        """
        Extracts answers from a list of Wikipedia page dictionaries.

        Iterates through each page, calls `extract_answer` for its text content,
        and collects all found answers along with their scores and source page titles.
        The results are sorted by confidence score in descending order.

        Args:
            question (str): The question string.
            wikipedia_pages (list[dict]): A list of dictionaries, where each
                                          dictionary represents a Wikipedia page and
                                          must contain at least "text" and "title" keys.
                                          Example: `[{"id": "...", "title": "...", "text": "..."}, ...]`

        Returns:
            list[dict]: A list of candidate answers, each as a dictionary:
                        `{"answer": str, "score": float, "source_page_title": str}`.
                        The list is sorted by score in descending order. Returns an
                        empty list if no answers are found or if input is invalid.
        """
        if not isinstance(question, str) or not question.strip():
            print("Warning: extract_answers_from_pages received empty or invalid question. Returning empty list.")
            return []
        if not isinstance(wikipedia_pages, list):
            print("Warning: extract_answers_from_pages received non-list input for wikipedia_pages. Returning empty list.")
            return []

        candidate_answers = []
        for page in wikipedia_pages:
            if not isinstance(page, dict) or "text" not in page or "title" not in page:
                print(f"Warning: Skipping invalid page data in extract_answers_from_pages: {page}")
                continue

            page_text = page.get("text", "")
            page_title = page.get("title", "Unknown Title")

            if not isinstance(page_text, str) or not page_text.strip():
                # print(f"Warning: Skipping page '{page_title}' with empty or invalid text content.")
                continue # Silently skip pages with no text

            answer, score = self.extract_answer(question, page_text)
            if answer: # Ensure answer is not None and not an empty string (though pipeline usually returns None for no answer)
                candidate_answers.append({
                    "answer": answer,
                    "score": score,
                    "source_page_title": page_title
                })

        # Sort by score for easier processing in answer_selector
        candidate_answers.sort(key=lambda x: x['score'], reverse=True)
        return candidate_answers

if __name__ == "__main__":
    reader = QAModelReader()

    sample_question = "What is the main function of insulin?"
    sample_context_1 = "Insulin is a peptide hormone produced by beta cells of the pancreatic islets; it is considered to be the main anabolic hormone of the body. It regulates the metabolism of carbohydrates, fats and protein by promoting the absorption of glucose from the blood into liver, fat and skeletal muscle cells."
    sample_context_2 = "The pancreas is a glandular organ in the digestive system and endocrine system of vertebrates. In humans, it is located in the abdomen behind the stomach."

    candidate_pages = [
        {"id": "Insulin", "title": "Insulin", "text": sample_context_1},
        {"id": "Pancreas", "title": "Pancreas", "text": sample_context_2}
    ]

    extracted_answers = reader.extract_answers_from_pages(sample_question, candidate_pages)

    print(f"\nExtracted answers for '{sample_question}':")
    for ans in extracted_answers:
        print(f"  - Answer: '{ans['answer']}' (Score: {ans['score']:.4f}) from '{ans['source_page_title']}'")