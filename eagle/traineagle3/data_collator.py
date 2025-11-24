"""Data collator for EAGLE-3 training."""

from typing import List, Dict, Any
import torch


class DataCollatorWithPadding:
    """
    Collator that pads sequences to the maximum length in the batch.
    """
    
    def __init__(self, pad_token_id: int = 0):
        """
        Args:
            pad_token_id: Token ID to use for padding
        """
        self.pad_token_id = pad_token_id
    
    def _pad_tensor_2d(self, tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Pad a 2D tensor to target length.
        
        Args:
            tensor: Input tensor of shape (batch_size, seq_len)
            target_length: Desired sequence length
            
        Returns:
            Padded tensor of shape (batch_size, target_length)
        """
        batch_size, current_length = tensor.shape
        padding = torch.zeros(batch_size, target_length - current_length, dtype=tensor.dtype)
        return torch.cat((tensor, padding), dim=1)
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features by padding to maximum length.
        
        Args:
            features: List of feature dictionaries
            
        Returns:
            Batched and padded tensors
        """
        max_length = max(item['input_ids'].shape[1] for item in features)
        
        batch_input_ids = torch.cat([
            self._pad_tensor_2d(item['input_ids'], max_length) 
            for item in features
        ])
        batch_attention_mask = torch.cat([
            self._pad_tensor_2d(item['attention_mask'], max_length) 
            for item in features
        ])
        batch_loss_mask = torch.cat([
            self._pad_tensor_2d(item['loss_mask'], max_length) 
            for item in features
        ])
        
        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
