import logging
import os
import random
import numpy as np
import torch
import config

def setup_logging(level=logging.INFO):
    """Sets up basic logging."""
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_dir(directory_path: str):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")

def set_seed(seed: int = config.RANDOM_SEED):
    """Sets random seed for reproducibility."""
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