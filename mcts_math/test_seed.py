import os
import random
from datetime import datetime
from typing import Any, Dict, List, Tuple
import torch
import numpy as np
from test_seed2 import get_value_estimate

def set_seed(seed: int = 1024) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # logger.info(f"Random seed set as {seed}")

def main():
    set_seed(42)
    value_estimate = random.random()
    print(value_estimate)
    print("Seed set successfully")
    value_get = get_value_estimate()
    print(value_get)

if __name__ == "__main__":
    main()

