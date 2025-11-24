"""Data preprocessing utilities for EAGLE-3 training."""

from typing import Dict, Any, List
import torch
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset

from constants import SYSTEM_PROMPT, ASSISTANT_SEPARATOR, USER_SEPARATOR, ROLE_MAPPING, CONV_ROLES


def build_messages(conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Build messages list from conversation data.
    
    Args:
        conversations: List of conversation turns with 'from' and 'value' keys
        
    Returns:
        List of messages with 'role' and 'content' keys
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if not conversations:
        return messages
        
    # Skip first message if it's not from human
    start_idx = 1 if ROLE_MAPPING.get(conversations[0]["from"]) != "user" else 0
    
    for j, sentence in enumerate(conversations[start_idx:]):
        role = ROLE_MAPPING[sentence["from"]]
        assert role == CONV_ROLES[j % 2], f"Unexpected role sequence at position {j}"
        messages.append({"role": role, "content": sentence["value"]})
        
    return messages


def compute_loss_mask(conversation: str, input_ids: torch.Tensor, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    """
    Compute loss mask for supervised fine-tuning.
    Only compute loss on assistant responses, not user instructions.
    
    Args:
        conversation: Full conversation string
        input_ids: Tokenized input IDs
        tokenizer: Tokenizer used for encoding
        
    Returns:
        Binary mask indicating which tokens to include in loss
    """
    loss_mask = torch.ones_like(input_ids)
    
    # Split conversation into turns
    turns = conversation.split(USER_SEPARATOR)
    turns[1] = turns[0] + USER_SEPARATOR + turns[1]
    turns = turns[1:]
    
    cur_len = 1
    loss_mask[:cur_len] = 0
    
    for i, turn in enumerate(turns):
        if turn == "":
            break
            
        turn_len = len(tokenizer(turn).input_ids)
        parts = turn.split(ASSISTANT_SEPARATOR)
        
        if len(parts) != 2:
            break
            
        parts[0] += ASSISTANT_SEPARATOR
        # "-2" is hardcoded for the Llama tokenizer to make the offset correct
        instruction_len = len(tokenizer(parts[0]).input_ids) - 1
        
        # Mask out user instructions
        if i == 0:
            loss_mask[cur_len: cur_len + instruction_len - 2] = 0
        else:
            loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
            
        cur_len += turn_len
        if i != 0:
            cur_len += 3
            
    loss_mask[cur_len:] = 0
    return loss_mask


def preprocess_function(examples: Dict[str, Any], tokenizer: PreTrainedTokenizerBase, max_len: int) -> Dict[str, Any]:
    """
    Preprocess examples by tokenizing and creating loss masks.
    
    Args:
        examples: Batch of examples with 'id' and 'conversations' keys
        tokenizer: Tokenizer for encoding
        max_len: Maximum sequence length
        
    Returns:
        Dictionary with input_ids, attention_mask, and loss_mask
    """
    new_examples = {
        "attention_mask": [],
        "input_ids": [],
        "loss_mask": []
    }
    
    for i in range(len(examples['id'])):
        conversations = examples['conversations'][i]
        if not conversations:
            continue
            
        try:
            messages = build_messages(conversations)
        except (KeyError, AssertionError):
            continue
            
        conversation = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.unk_token_id
            
        input_ids = tokenizer(
            conversation,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids[0]
        
        # Filter out samples longer than max_len
        if len(input_ids) > max_len:
            continue
            
        loss_mask = compute_loss_mask(conversation, input_ids, tokenizer)
        attention_mask = torch.ones_like(loss_mask)
        
        new_examples["input_ids"].append(input_ids[None, :])
        new_examples["loss_mask"].append(loss_mask[None, :])
        new_examples["attention_mask"].append(attention_mask[None, :])
        
    return new_examples


def build_dataset_rank(tokenizer: PreTrainedTokenizerBase, datapath: str, max_len: int, num_proc: int = 8):
    """
    Build and preprocess dataset from JSON file.
    
    Args:
        tokenizer: Tokenizer for encoding
        datapath: Path to JSONL data file
        max_len: Maximum sequence length
        num_proc: Number of processes for parallel preprocessing
        
    Returns:
        Preprocessed dataset in torch format
    """
    ds = load_dataset('json', data_files=datapath)
    ds = ds['train'].shuffle(seed=42)
    original_columns = ds.column_names
    
    ds = ds.map(
        lambda examples: preprocess_function(examples, tokenizer, max_len),
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
        load_from_cache_file=False
    )
    
    ds.set_format(type="torch")
    return ds
