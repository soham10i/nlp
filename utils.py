import logging
import os
import random
import numpy as np
import torch
# import config # config is not used directly for RANDOM_SEED anymore

def setup_logging(level: int = logging.INFO):
    """
    Sets up basic logging for the application.

    Args:
        level (int, optional): The logging level (e.g., logging.INFO, logging.DEBUG).
                               Defaults to logging.INFO.
    """
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def ensure_dir(directory_path: str):
    """
    Ensures that a directory exists at the specified path.
    If the directory (or any parent directory) does not exist, it is created.

    Args:
        directory_path (str): The path to the directory.
    """
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            logging.info(f"Created directory: {directory_path}")
        except OSError as e:
            logging.error(f"Error creating directory {directory_path}: {e}")
            # Depending on severity, might re-raise or handle differently
            raise # Re-raise the exception if directory creation is critical

def set_seed(seed: int = 42):
    """
    Sets random seeds for Python's `random`, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int, optional): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

# Add any other common utility functions here, e.g., for file I/O, text cleaning, etc.

if __name__ == '__main__':
    setup_logging()
    ensure_dir("test_dir")
    logging.info("Utils test complete.")
    os.rmdir("test_dir")