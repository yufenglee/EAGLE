"""Constants used in EAGLE-3 training."""

# System prompt for chat template
SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
    "while being safe.  Your answers should not include any harmful, unethical, racist, sexist, "
    "toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased "
    "and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why instead of "
    "answering something not correct. If you don't know the answer to a question, please don't "
    "share false information."
)

# Chat template separators
ASSISTANT_SEPARATOR = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
USER_SEPARATOR = "<|eot_id|><|start_header_id|>user<|end_header_id|>"

# Role mappings
ROLE_MAPPING = {"human": "user", "gpt": "assistant"}
CONV_ROLES = ["user", "assistant"]

# Training configuration defaults
DEFAULT_NUM_EPOCHS = 40
DEFAULT_NUM_WORKERS = 2
DEFAULT_MAX_LEN = 2048
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRADIENT_CHECKPOINT = True

# Loss weight decay factor for multiple prediction heads
LOSS_WEIGHT_DECAY = 0.8
