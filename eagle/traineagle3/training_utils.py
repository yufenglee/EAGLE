"""Training loop utilities for EAGLE-3 training."""

from typing import List, Dict, Optional
import torch
import deepspeed
from tqdm import tqdm

from constants import LOSS_WEIGHT_DECAY


def compute_weighted_loss(losses: List[torch.Tensor], weight_decay: float = LOSS_WEIGHT_DECAY) -> torch.Tensor:
    """
    Compute weighted sum of losses with exponential decay.
    
    Args:
        losses: List of loss tensors
        weight_decay: Decay factor for loss weights
        
    Returns:
        Weighted sum of losses
    """
    weights = [weight_decay ** i for i in range(len(losses))]
    return sum(w * loss for w, loss in zip(weights, losses))


def train_epoch(
    model_engine,
    train_loader,
    optimizer,
    rank: int,
    global_rank: int,
    logger=None
) -> tuple:
    """
    Run one training epoch.
    
    Args:
        model_engine: DeepSpeed model engine
        train_loader: Training data loader
        optimizer: Optimizer
        rank: Local rank
        global_rank: Global rank for distributed training
        logger: Optional wandb logger
        
    Returns:
        Tuple of (epoch_accuracies, epoch_losses) aggregated over all batches
    """
    model_engine.module.train()
    epoch_accs = [[] for _ in range(model_engine.module.length)]
    epoch_plosses = [[] for _ in range(model_engine.module.length)]
    
    for batch_idx, data in enumerate(tqdm(train_loader, disable=global_rank != 0)):
        model_engine.zero_grad()
        
        plosses, vlosses, acces = model_engine(
            input_ids=data["input_ids"].to(rank),
            attention_mask=data["attention_mask"].to(rank),
            loss_mask=data["loss_mask"],
        )
        
        loss = compute_weighted_loss(plosses)
        model_engine.backward(loss)
        model_engine.step()
        
        # Log training metrics
        if global_rank == 0 and logger is not None:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            logger.log(logdict)
        
        # Accumulate epoch statistics
        for i in range(len(acces)):
            epoch_accs[i].append(acces[i])
        for i in range(len(plosses)):
            epoch_plosses[i].append(plosses[i].item())
    
    return epoch_accs, epoch_plosses


def evaluate_epoch(
    model_engine,
    test_loader,
    rank: int,
    global_rank: int
) -> tuple:
    """
    Run one evaluation epoch.
    
    Args:
        model_engine: DeepSpeed model engine
        test_loader: Test data loader
        rank: Local rank
        global_rank: Global rank for distributed training
        
    Returns:
        Tuple of (epoch_accuracies, epoch_losses) aggregated over all batches
    """
    model_engine.module.eval()
    epoch_accs = [[] for _ in range(model_engine.module.length)]
    epoch_plosses = [[] for _ in range(model_engine.module.length)]
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader, disable=global_rank != 0)):
            plosses, vlosses, acces = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"],
            )
            
            # Accumulate epoch statistics
            for i in range(len(acces)):
                epoch_accs[i].append(acces[i])
            for i in range(len(plosses)):
                epoch_plosses[i].append(plosses[i].item())
    
    return epoch_accs, epoch_plosses


def aggregate_and_log_metrics(
    epoch_metrics: List[List[float]],
    metric_name: str,
    epoch: int,
    num_epochs: int,
    global_rank: int,
    logger=None,
    log_prefix: str = "train"
) -> None:
    """
    Aggregate metrics across ranks and log them.
    
    Args:
        epoch_metrics: List of metric values for each position
        metric_name: Name of the metric (e.g., 'acc', 'ploss')
        epoch: Current epoch number
        num_epochs: Total number of epochs
        global_rank: Global rank for distributed training
        logger: Optional wandb logger
        log_prefix: Prefix for logging (e.g., 'train' or 'test')
    """
    for i in range(len(epoch_metrics)):
        metric_value = torch.tensor(epoch_metrics[i]).cuda().mean()
        deepspeed.comm.all_reduce(metric_value, op=deepspeed.comm.ReduceOp.AVG)
        metric_value = metric_value.item()
        
        if global_rank == 0:
            print(f"{log_prefix.capitalize()} Epoch [{epoch + 1}/{num_epochs}], "
                  f"position {i}, {metric_name}: {metric_value:.2f}")
            if logger is not None:
                logger.log({f"{log_prefix}/epoch{metric_name}_{i}": metric_value})
