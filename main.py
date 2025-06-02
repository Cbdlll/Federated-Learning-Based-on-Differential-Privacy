#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import random

from server import FederatedServer
from utils import setup_logger

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser(description='Federated Learning with LeNet and Differential Privacy')
    parser.add_argument('--num_rounds', type=int, default=100, help='Number of communication rounds')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--batch_size', type=int, default=64, help='Local batch size')
    parser.add_argument('--local_epochs', type=int, default=5, help='Number of local epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--dp_epsilon', type=float, default=1.0, help='Differential privacy epsilon parameter')
    parser.add_argument('--dp_delta', type=float, default=1e-5, help='Differential privacy delta parameter')
    parser.add_argument('--dp_max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for DP')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    parser.add_argument('--log_interval', type=int, default=1, help='Log interval')
    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logger("federated_learning")
    logger.info(f"Starting Federated Learning with {args.num_clients} clients")
    logger.info(f"Using device: {args.device}")
    
    # Initialize server
    server = FederatedServer(num_clients=args.num_clients, 
                             local_batch_size=args.batch_size,
                             local_epochs=args.local_epochs,
                             learning_rate=args.lr,
                             momentum=args.momentum,
                             epsilon=args.dp_epsilon,
                             delta=args.dp_delta,
                             max_grad_norm=args.dp_max_grad_norm,
                             device=args.device,
                             logger=logger)
    
    # Run federated learning
    server.run_federated_learning(num_rounds=args.num_rounds, log_interval=args.log_interval)

if __name__ == "__main__":
    main() 