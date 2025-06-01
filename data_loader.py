from datasets import load_dataset
import os
import json
from config import MEDQA_DATASET_NAME

def load_medqa_dataset(split="train"):
    """
    Loads the MedQA-USMLE-4-options dataset from Hugging Face. [cite: 130]
    """
    print(f"Loading MedQA dataset (split: {split})...")
    dataset = load_dataset(MEDQA_DATASET_NAME, split=split) 
    print(f"Dataset loaded. Number of examples: {len(dataset)}")
    return dataset

# No load_wikipedia_data() function here anymore, as it's fetched on demand.

if __name__ == "__main__":
    # Example usage:
    train_data = load_medqa_dataset(split="train")
    test_data = load_medqa_dataset(split="test")

    print("\nFirst MedQA Training Question:")
    print(f"ID: {train_data[0]['id']}") 
    print(f"Question: {train_data[0]['sent1']}") 
    print(f"Options: {train_data[0]['ending0']}, {train_data[0]['ending1']}, {train_data[0]['ending2']}, {train_data[0]['ending3']}") 
    print(f"Correct Label Index: {train_data[0]['label']}") 
