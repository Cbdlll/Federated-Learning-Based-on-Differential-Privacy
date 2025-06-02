#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import copy
import numpy as np

from models.lenet import LeNet
from client import Client
from data_loader import load_and_split_data

class FederatedServer:
    """Server for federated learning"""
    def __init__(self, num_clients=10, local_batch_size=64, local_epochs=5,
                 learning_rate=0.01, momentum=0.9, epsilon=1.0, delta=1e-5,
                 max_grad_norm=1.0, device='cpu', logger=None):
        self.num_clients = num_clients
        self.local_batch_size = local_batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.logger = logger
        
        # Initialize global model
        self.global_model = LeNet().to(self.device)
        
        # Load and split data
        train_datasets, test_datasets = load_and_split_data(num_clients)
        
        # Initialize clients
        self.clients = [
            Client(
                client_id=i,
                train_data=train_datasets[i],
                test_data=test_datasets[i],
                batch_size=local_batch_size,
                learning_rate=learning_rate,
                momentum=momentum,
                local_epochs=local_epochs,
                device=device
            )
            for i in range(num_clients)
        ]
        
        self.logger.info(f"Initialized server with {num_clients} clients")
    
    def aggregate_models(self, client_params):
        """Aggregate client models using FedAvg algorithm"""
        # Simple averaging of model parameters
        aggregated_params = []
        for param_idx in range(len(client_params[0])):
            aggregated_params.append(
                torch.stack([client_params[i][param_idx] for i in range(len(client_params))]).mean(dim=0)
            )
        
        # Update global model with aggregated parameters
        self.global_model.set_params(aggregated_params)
    
    def evaluate_global_model(self):
        """Evaluate global model on all client test data"""
        total_loss = 0
        total_accuracy = 0
        
        for client in self.clients:
            loss, accuracy = client.evaluate(self.global_model)
            total_loss += loss
            total_accuracy += accuracy
        
        avg_loss = total_loss / self.num_clients
        avg_accuracy = total_accuracy / self.num_clients
        
        return avg_loss, avg_accuracy
    
    def run_federated_learning(self, num_rounds=100, log_interval=1):
        """Run federated learning for specified number of rounds"""
        for round_idx in range(1, num_rounds + 1):
            self.logger.info(f"Round {round_idx}/{num_rounds}")
            
            # Select clients for this round (in this case, all clients)
            selected_clients = list(range(self.num_clients))
            
            # Collect updated parameters from clients
            client_params = []
            for client_idx in selected_clients:
                client = self.clients[client_idx]
                # Train client with differential privacy
                params = client.train(
                    self.global_model, 
                    use_dp=True,
                    epsilon=self.epsilon,
                    delta=self.delta,
                    max_grad_norm=self.max_grad_norm
                )
                client_params.append(params)
            
            # Aggregate client models
            self.aggregate_models(client_params)
            
            # Evaluate global model
            if round_idx % log_interval == 0:
                avg_loss, avg_accuracy = self.evaluate_global_model()
                self.logger.info(f"Round {round_idx}: Average Loss = {avg_loss:.4f}, Average Accuracy = {avg_accuracy:.4f}")
        
        # Final evaluation
        avg_loss, avg_accuracy = self.evaluate_global_model()
        self.logger.info(f"Final evaluation - Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")
        
        return self.global_model 