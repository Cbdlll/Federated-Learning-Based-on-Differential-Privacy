#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import copy
import numpy as np
import os
import json

from models.lenet import LeNet
from client import Client
from data_loader import load_and_split_data

class FederatedServer:
    """Server for federated learning"""
    def __init__(self, num_clients=10, local_batch_size=64, local_epochs=5,
                 learning_rate=0.01, momentum=0.9, epsilon=1.0, delta=1e-5,
                 max_grad_norm=1.0, noise_type='gaussian', device='cpu', logger=None):
        self.num_clients = num_clients
        self.local_batch_size = local_batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_type = noise_type
        self.device = device
        self.logger = logger
        
        # Initialize global model
        self.global_model = LeNet().to(self.device)
        
        # Load and split data
        train_datasets, test_datasets = load_and_split_data(num_clients, device=device)
        
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
                noise_type = noise_type,
                device=device
            )
            for i in range(num_clients)
        ]
        
        # Initialize accuracy tracking
        self.accuracy_history = {}
        
        self.logger.info(f"Initialized server with {num_clients} clients")
        if torch.cuda.is_available():
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    def aggregate_models(self, client_params):
        """Aggregate client models using FedAvg algorithm"""
        # Simple averaging of model parameters
        aggregated_params = []
        for param_idx in range(len(client_params[0])):
            # Move client parameters to server device for aggregation
            params_on_device = [client_params[i][param_idx].to(self.device) for i in range(len(client_params))]
            aggregated_params.append(
                torch.stack(params_on_device).mean(dim=0)
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
    
    def save_model(self, save_path='models/saved'):
        """Save the global model to disk
        
        Args:
            save_path: Directory to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        # Save the model
        model_path = os.path.join(save_path, f'{self.noise_type}_{self.epsilon}_federated_model.pth')
        torch.save(self.global_model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")
        
        # Save model architecture information
        info_path = os.path.join(save_path, f'{self.noise_type}_{self.epsilon}_model_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Model Architecture: LeNet\n")
            f.write(f"Number of clients: {self.num_clients}\n")
            f.write(f"Training Parameters:\n")
            f.write(f"  Batch size: {self.local_batch_size}\n")
            f.write(f"  Local epochs: {self.local_epochs}\n")
            f.write(f"  Learning rate: {self.learning_rate}\n")
            f.write(f"  Momentum: {self.momentum}\n")
            f.write(f"  DP epsilon: {self.epsilon}\n")
            f.write(f"  DP delta: {self.delta}\n")
            f.write(f"  DP max gradient norm: {self.max_grad_norm}\n")
        
        return model_path
    
    def save_accuracy_history(self, save_path='results'):
        """Save accuracy history to a JSON file with append mode
        
        Args:
            save_path: Directory to save the accuracy history
        """
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save path for accuracy history
        history_path = os.path.join(save_path, f'{self.noise_type}_accuracy_history.json')
        
        # Read existing data if file exists
        existing_history = {}
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                existing_history = json.load(f)
        
        # Merge new accuracy history with existing data
        for epsilon, data in self.accuracy_history.items():
            if epsilon not in existing_history:
                existing_history[epsilon] = data
            else:
                # Append new rounds and accuracies
                existing_history[epsilon]['rounds'].extend(data['rounds'])
                existing_history[epsilon]['accuracies'].extend(data['accuracies'])
        
        # Save merged history
        with open(history_path, 'w') as f:
            json.dump(existing_history, f, indent=4)
        
        self.logger.info(f"Accuracy history appended to {history_path}")
        return history_path
    
    def run_federated_learning(self, num_rounds=100, log_interval=1, use_dp=False, epsilon_values=None):
        """Run federated learning for specified number of rounds
        
        Args:
            num_rounds: Number of federated learning rounds
            log_interval: Interval for logging results
            epsilon_values: List of epsilon values to test
        """
        # If no epsilon values provided, use the default epsilon
        if epsilon_values is None:
            epsilon_values = [self.epsilon]
        
        # Iterate through different epsilon values
        for epsilon in epsilon_values:
            # Reset the global model for each epsilon
            self.global_model = LeNet().to(self.device)
            
            # Set the current epsilon
            self.epsilon = epsilon
            
            # Initialize accuracy tracking for this epsilon
            self.accuracy_history[str(epsilon)] = {
                'rounds': [],
                'accuracies': []
            }
            
            self.logger.info(f"Training with epsilon = {epsilon}")
            
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
                        use_dp=use_dp,
                        epsilon=epsilon,
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
                    
                    # Store accuracy for this round
                    self.accuracy_history[str(epsilon)]['rounds'].append(round_idx)
                    self.accuracy_history[str(epsilon)]['accuracies'].append(avg_accuracy)
            
            # Final evaluation
            avg_loss, avg_accuracy = self.evaluate_global_model()
            self.logger.info(f"Final evaluation for epsilon {epsilon} - Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")
            
            # Save the final model for this epsilon
            model_path = self.save_model()
            self.logger.info(f"Training complete for epsilon {epsilon}. Final model saved to {model_path}")
        
        # Save accuracy history
        self.save_accuracy_history()
        
        return self.global_model 