# EAGLE-3 Training Code Refactoring

This directory contains the refactored training code for EAGLE-3, which has been reorganized for better readability and maintainability.

## Structure

The training code has been split into the following modules:

### Core Files

- **`main.py`** - Main training script (refactored)
  - Clean, well-documented entry point
  - Separated concerns with clear functions
  - Uses utility modules for better organization

- **`cnets.py`** - Model architecture and neural network components
  - Contains the EAGLE-3 model definition
  - Attention mechanisms and decoder layers
  - Vocabulary scanning for draft model

- **`configs.py`** - Model configuration classes
  - Configuration dataclasses for model setup

- **`modeling_llama_kv.py`** - LLaMA base model with KV cache support

### Utility Modules (New)

- **`constants.py`** - Centralized constants
  - System prompts
  - Chat template separators
  - Role mappings
  - Training hyperparameters

- **`data_utils.py`** - Data preprocessing utilities
  - Dataset building and loading
  - Message formatting
  - Loss mask computation
  - Tokenization logic

- **`data_collator.py`** - Data collation for batching
  - Padding sequences to uniform length
  - Creating batched tensors

- **`checkpoint_utils.py`** - Checkpoint management
  - Finding latest checkpoints
  - Resume training from checkpoints

- **`training_utils.py`** - Training and evaluation loops
  - Training epoch logic
  - Evaluation epoch logic
  - Metric aggregation and logging
  - Loss computation with weighted decay

## Key Improvements

### 1. Separation of Concerns
- Data preprocessing separated from training logic
- Checkpoint management isolated into its own module
- Training/evaluation loops extracted into reusable functions

### 2. Reduced Code Duplication
- Single source of truth for preprocessing logic
- Shared constants across modules
- Unified data collation

### 3. Better Readability
- Clear function names and docstrings
- Logical organization by purpose
- Reduced file length (main.py: 352 lines → 277 lines)

### 4. Improved Maintainability
- Constants defined in one place
- Easier to test individual components
- Better error handling

### 5. Enhanced Documentation
- Comprehensive docstrings
- Type hints where applicable
- Inline comments for complex logic

## Usage

The training script can be run with the same arguments as before:

```bash
deepspeed --include localhost:0,1,2,3,4,5,6,7 main.py \
    --deepspeed_config ds_config.json \
    --basepath /path/to/base/model \
    --trainpath /path/to/train.jsonl \
    --testpath /path/to/test.json \
    --savedir ./checkpoints
```

## Configuration Files

- **`config.json`** - Model architecture configuration
- **`ds_config.json`** - DeepSpeed configuration for distributed training
- **`default_config.yaml`** - Default training configuration

## Backward Compatibility

The refactored code maintains full backward compatibility:
- Same command-line arguments
- Same checkpoint format
- Same training behavior
- Original `main.py` preserved as `main_old.py`

## Migration Notes

If you have custom modifications to the old `main.py`, you can:
1. Review `main_old.py` for your changes
2. Apply them to the appropriate new module
3. The modular structure should make this easier

## Future Improvements

Potential areas for further enhancement:
- Add configuration file for training parameters
- Implement automatic mixed precision (AMP) wrapper
- Add validation checks for hyperparameters
- Create unit tests for utility functions
- Add profiling utilities
