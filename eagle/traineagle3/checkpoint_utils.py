"""Checkpoint management utilities for EAGLE-3 training."""

import os
import re
from typing import Optional, Tuple


def find_latest_checkpoint(directory: str, filename: str = "zero_to_fp32.py") -> Tuple[Optional[str], int]:
    """
    Find the latest checkpoint in a directory based on state number.
    
    Args:
        directory: Directory containing checkpoint subdirectories
        filename: Filename to verify checkpoint validity (default: zero_to_fp32.py)
        
    Returns:
        Tuple of (checkpoint_path, next_epoch) where checkpoint_path is None if no checkpoint found
    """
    if not os.path.exists(directory):
        return None, 0
        
    max_state = -1
    
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            state_num = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_state = max(max_state, state_num)
    
    if max_state == -1:
        return None, 0
        
    return f"{directory}/state_{max_state}", max_state + 1
