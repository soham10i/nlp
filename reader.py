from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
from config import QA_MODEL_NAME, QA_MODEL_PATH, DEVICE

class QAModelReader:
    def __init__(self):
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
        print("QA model loaded.")

    def extract_answer(self, question, context):
        """
        Extracts an answer span from the given context for the question. [cite: 217]
        Returns the answer text, start, end, and confidence score.
        """
        try:
            # The pipeline handles tokenization, model inference, and span extraction
            result = self.qa_pipeline(question=question, context=context) 
            # result structure: {'score': float, 'start': int, 'end': int, 'answer': str}
            return result['answer'], result['score'] 
        except Exception as e:
            print(f"Error during QA extraction for question '{question[:50]}...' and context '{context[:50]}...': {e}")
            return None, 0.0

    def extract_answers_from_pages(self, question, wikipedia_pages):
        """
        Extracts answers from a list of Wikipedia page dictionaries. [cite: 215]
        wikipedia_pages: List of {"id": ..., "title": ..., "text": ...}
        Returns a list of candidate answers with scores, e.g.,
        [{"answer": "...", "score": ..., "source_page_title": "..."}, ...]
        """
        candidate_answers = []
        for page in wikipedia_pages:
            answer, score = self.extract_answer(question, page["text"])
            if answer:
                candidate_answers.append({
                    "answer": answer,
                    "score": score,
                    "source_page_title": page["title"]
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