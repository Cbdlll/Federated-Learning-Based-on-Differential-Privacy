#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import random
import json
import os

from server import FederatedServer
from models.lenet import LeNet
from utils import setup_logger

def parse_float_with_inf(value):
    return float('inf') if value.lower() == 'inf' else float(value)

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_model(model_path, device='cpu'):
    """Load a saved model for inference
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    model = LeNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def main():
    parser = argparse.ArgumentParser(description='Federated Learning with LeNet and Differential Privacy')
    parser.add_argument('--num_rounds', type=int, default=30, help='Number of communication rounds')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--batch_size', type=int, default=256, help='Local batch size')
    parser.add_argument('--local_epochs', type=int, default=2, help='Number of local epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--use_dp', type=bool, default=False, help='Use differential privacy')
    parser.add_argument('--dp_epsilon', type=float, default=10.0, help='Differential privacy epsilon parameter')
    parser.add_argument('--dp_epsilon_values', type=parse_float_with_inf, nargs='+', default=None, help='List of epsilon values to test')
    parser.add_argument('--dp_delta', type=float, default=1e-5, help='Differential privacy delta parameter')
    parser.add_argument('--dp_max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for DP')
    parser.add_argument('--noise_type', type=str, default='gaussian', help='Noise type for DP')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=1, help='Log interval')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='models/saved', help='Directory to save the model')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a saved model to load (for inference)')
    args = parser.parse_args()

    # Configure GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Using CPU")

    # If loading a model for inference
    if args.load_model:
        model = load_model(args.load_model, device)
        print("Model loaded for inference. You can now use it for predictions.")
        return

    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logger("federated_learning")
    logger.info(f"Starting Federated Learning with {args.num_clients} clients")
    logger.info(f"Using device: {device}")
    
    # Initialize server
    server = FederatedServer(num_clients=args.num_clients, 
                             local_batch_size=args.batch_size,
                             local_epochs=args.local_epochs,
                             learning_rate=args.lr,
                             momentum=args.momentum,
                             epsilon=args.dp_epsilon,
                             delta=args.dp_delta,
                             max_grad_norm=args.dp_max_grad_norm,
                             noise_type=args.noise_type,
                             device=device,
                             logger=logger)
    
    # Run federated learning
    final_model = server.run_federated_learning(num_rounds=args.num_rounds, use_dp=args.use_dp, log_interval=args.log_interval, epsilon_values=args.dp_epsilon_values)

if __name__ == "__main__":
    main() 