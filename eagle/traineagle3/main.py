"""
EAGLE-3 Training Script

This script trains the EAGLE-3 model using DeepSpeed for distributed training.
The code has been refactored for better readability and maintainability.
"""

import argparse
import json
import os

import deepspeed
import torch
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from cnets import Model
from configs import EConfig
from constants import (
    LOSS_WEIGHT_DECAY,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_NUM_PROC,
    DEFAULT_MAX_LEN,
    DEFAULT_GRADIENT_CHECKPOINT,
    SYSTEM_PROMPT,
    ASSISTANT_SEPARATOR,
    USER_SEPARATOR,
    ROLE_MAPPING,
    CONV_ROLES
)
from data_utils import build_dataset_rank
from data_collator import DataCollatorWithPadding
from checkpoint_utils import find_latest_checkpoint
from training_utils import train_epoch, evaluate_epoch, aggregate_and_log_metrics


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='EAGLE-3 Training')
    parser.add_argument('--basepath', type=str, 
                        default='/home/lyh/weights/hf/llama31chat/8B/',
                        help='Path to base model weights')
    parser.add_argument('--trainpath', type=str,
                        default="/home/lyh/code/nlp/developing/vllmbase/vllm/gedata/l318b.jsonl",
                        help='Path to training data')
    parser.add_argument('--testpath', type=str,
                        default="/home/lyh/code/nlp/developing/vllmbase/vllm/gedata/0318.json",
                        help='Path to test data')
    parser.add_argument('--savedir', type=str, default='0',
                        help='Directory to save checkpoints')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def load_training_config(deepspeed_config_path: str) -> dict:
    """
    Load training configuration from DeepSpeed config file.
    
    Args:
        deepspeed_config_path: Path to DeepSpeed configuration JSON
        
    Returns:
        Dictionary with training configuration
    """
    with open(deepspeed_config_path) as f:
        ds_config = json.load(f)
    
    return {
        "bs": ds_config["train_micro_batch_size_per_gpu"],
        "num_epochs": DEFAULT_NUM_EPOCHS,
        "num_workers": DEFAULT_NUM_WORKERS,
        "max_len": DEFAULT_MAX_LEN,
        "config_path": "config.json",
        "gradient_checkpoint": DEFAULT_GRADIENT_CHECKPOINT
    }


def setup_distributed():
    """
    Setup distributed training environment.
    
    Returns:
        Tuple of (global_rank, local_rank, world_size)
    """
    global_rank = deepspeed.comm.get_rank()
    local_rank = deepspeed.comm.get_local_rank()
    world_size = deepspeed.comm.get_world_size()
    return global_rank, local_rank, world_size


def setup_wandb(global_rank: int, ds_config: dict):
    """
    Setup Weights & Biases logging for rank 0.
    
    Args:
        global_rank: Global rank of current process
        ds_config: DeepSpeed configuration
        
    Returns:
        wandb module if rank 0, else None
    """
    if global_rank == 0:
        import wandb
        wandb.login(key="")
        wandb.init(project="l382", entity="yuhui-li", config=ds_config)
        return wandb
    return None


def create_data_loaders(
    train_dataset,
    test_dataset,
    batch_size: int,
    world_size: int,
    global_rank: int,
    num_workers: int = 4
):
    """
    Create training and test data loaders with distributed sampling.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size per GPU
        world_size: Total number of GPUs
        global_rank: Global rank of current process
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    collate_fn = DataCollatorWithPadding()
    
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=global_rank, 
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader


def save_checkpoint(model_engine, save_dir: str, epoch: int, global_rank: int):
    """
    Save model checkpoint.
    
    Args:
        model_engine: DeepSpeed model engine
        save_dir: Directory to save checkpoint
        epoch: Current epoch number
        global_rank: Global rank of current process
    """
    checkpoint_dir = f"{save_dir}/state_{epoch}"
    model_engine.save_16bit_model(checkpoint_dir, exclude_frozen_parameters=True)
    
    # Save full DeepSpeed checkpoint every 10 epochs
    if epoch % 10 == 0:
        deepspeed.DeepSpeedEngine.save_checkpoint(model_engine, save_dir=checkpoint_dir)
    
    if global_rank == 0:
        print(f"Checkpoint saved to {checkpoint_dir}")


def main():
    """Main training function."""
    # Setup
    args = parse_arguments()
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(0)
    
    # Load configurations
    train_config = load_training_config(args.deepspeed_config)
    with open(args.deepspeed_config) as f:
        ds_config = json.load(f)
    
    # Setup distributed training
    global_rank, local_rank, world_size = setup_distributed()
    
    # Setup logging
    wandb_logger = setup_wandb(global_rank, ds_config)
    
    # Create save directory
    os.makedirs(args.savedir, exist_ok=True)
    
    # Load tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(args.basepath)
    
    print(f"Loading training dataset from {args.trainpath}")
    train_dataset = build_dataset_rank(tokenizer, args.trainpath, train_config["max_len"], num_proc=DEFAULT_NUM_PROC)
    
    print(f"Loading test dataset from {args.testpath}")
    test_dataset = build_dataset_rank(tokenizer, args.testpath, train_config["max_len"], num_proc=DEFAULT_NUM_PROC)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_dataset,
        test_dataset,
        train_config["bs"],
        world_size,
        global_rank,
        train_config["num_workers"]
    )
    
    # Initialize model
    config = EConfig.from_pretrained(train_config["config_path"])
    model = Model(config, ds_config, train_config, path=args.basepath, load_emb=True, load_head=True)
    model.scandata(args.trainpath, args.basepath)
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
    )
    
    # Load checkpoint if exists
    checkpoint_path, start_epoch = find_latest_checkpoint(args.savedir)
    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model_engine.load_checkpoint(checkpoint_path)
    else:
        start_epoch = 0
        print("Starting training from scratch")
    
    # Training loop
    num_epochs = train_config["num_epochs"]
    for epoch in range(start_epoch, num_epochs):
        train_loader.sampler.set_epoch(epoch + 1)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training phase
        print("\nTraining...")
        epoch_accs, epoch_plosses = train_epoch(
            model_engine, train_loader, optimizer, local_rank, global_rank, wandb_logger
        )
        
        # Log training metrics
        aggregate_and_log_metrics(
            epoch_accs, "acc", epoch, num_epochs, global_rank, wandb_logger, "train"
        )
        aggregate_and_log_metrics(
            epoch_plosses, "ploss", epoch, num_epochs, global_rank, wandb_logger, "train"
        )
        
        # Evaluation phase
        print("\nEvaluating...")
        epoch_accs, epoch_plosses = evaluate_epoch(
            model_engine, test_loader, local_rank, global_rank
        )
        
        # Log evaluation metrics
        aggregate_and_log_metrics(
            epoch_accs, "acc", epoch, num_epochs, global_rank, wandb_logger, "test"
        )
        aggregate_and_log_metrics(
            epoch_plosses, "ploss", epoch, num_epochs, global_rank, wandb_logger, "test"
        )
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Save checkpoint
        save_checkpoint(model_engine, args.savedir, epoch, global_rank)
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    main()
