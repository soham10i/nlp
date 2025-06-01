from datasets import load_dataset
import os
import json
from config import MEDQA_DATASET_NAME
from datasets import Dataset # For type hinting

def load_medqa_dataset(split: str = "train") -> Dataset | None:
    """
    Loads a specified split of the MedQA-USMLE-4-options dataset from Hugging Face.

    Args:
        split (str, optional): The dataset split to load (e.g., "train", "test",
                               "validation"). Defaults to "train".

    Returns:
        datasets.Dataset | None: The loaded Hugging Face Dataset object, or None
                                 if loading fails.
    """
    print(f"Loading MedQA dataset (split: {split})...")
    try:
        dataset = load_dataset(MEDQA_DATASET_NAME, name="US", split=split) # Added name="US" as per HF dataset viewer for this dataset
        print(f"Dataset '{MEDQA_DATASET_NAME}' (split: {split}) loaded successfully. Number of examples: {len(dataset)}")
        return dataset
    except FileNotFoundError:
        print(f"Error: Dataset {MEDQA_DATASET_NAME} not found. It might be a configuration issue or the dataset was removed from Hugging Face Hub.")
        return None
    except ConnectionError:
        print(f"Error: Could not connect to Hugging Face Hub to download dataset {MEDQA_DATASET_NAME}. Please check your internet connection.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading dataset {MEDQA_DATASET_NAME} (split: {split}): {e}")
        return None

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
