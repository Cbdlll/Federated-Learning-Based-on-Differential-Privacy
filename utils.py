#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import torch
import json

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Set up logger with specified name and level
    
    Args:
        name: Name of the logger
        log_file: Log file path (optional)
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def save_model(model: torch.nn.Module, path: str):
    """
    Save model to disk
    
    Args:
        model: Model to save
        path: Path to save model to
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    
def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """
    Load model from disk
    
    Args:
        model: Model to load weights into
        path: Path to load model from
        
    Returns:
        Loaded model
    """
    model.load_state_dict(torch.load(path))
    return model

def save_metrics(metrics: Dict[str, List[Any]], path: str):
    """
    Save metrics to disk as JSON
    
    Args:
        metrics: Dictionary of metrics to save
        path: Path to save metrics to
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f)

def plot_metrics(metrics: Dict[str, List[float]], title: str, path: str):
    """
    Plot training metrics
    
    Args:
        metrics: Dictionary of metrics to plot
        title: Plot title
        path: Path to save plot to
    """
    plt.figure(figsize=(10, 6))
    
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)
    
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

def calculate_privacy_cost(batch_size: int, dataset_size: int, epochs: int, 
                          noise_multiplier: float, delta: float) -> float:
    """
    Calculate the privacy cost (epsilon) using moments accountant 
    (This is a simplified version - in practice use a proper library)
    
    Args:
        batch_size: Batch size used for training
        dataset_size: Size of the training dataset
        epochs: Number of epochs
        noise_multiplier: Noise multiplier
        delta: Target delta value
        
    Returns:
        Privacy budget (epsilon) spent
    """
    # Number of steps
    steps = epochs * dataset_size // batch_size
    
    # Sampling probability
    q = batch_size / dataset_size
    
    # Simplified calculation (in practice, use proper accountant)
    # This is just a rough approximation
    epsilon = np.sqrt(2 * np.log(1.25 / delta)) * q * np.sqrt(steps) / noise_multiplier
    
    return epsilon 