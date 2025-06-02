#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import copy

from models.lenet import LeNet
from privacy import PrivacyEngine

class Client:
    """Client for federated learning"""
    def __init__(self, client_id, train_data, test_data, batch_size, 
                 learning_rate, momentum, local_epochs, device):
        self.client_id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.local_epochs = local_epochs
        self.device = device
        
        # Initialize the model
        self.model = LeNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, global_model=None, use_dp=False, epsilon=1.0, delta=1e-5, max_grad_norm=1.0):
        """Train the local model with differential privacy if enabled"""
        # If global model provided, set local model parameters to global model parameters
        if global_model is not None:
            self.model.load_state_dict(copy.deepcopy(global_model.state_dict()))
        
        # Set up optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
        # Attach privacy engine if differential privacy is enabled
        if use_dp:
            privacy_engine = PrivacyEngine(
                self.model,
                batch_size=self.batch_size,
                sample_size=len(self.train_data),
                alphas=[1 + x / 10.0 for x in range(1, 100)],
                noise_multiplier=None,  # Will be computed from epsilon, delta
                max_grad_norm=max_grad_norm,
                epsilon=epsilon,
                delta=delta
            )
            privacy_engine.attach(optimizer)
        
        # Set model to training mode
        self.model.train()
        
        # Training loop
        train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )
        
        for epoch in range(self.local_epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                
                # Calculate loss
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
        
        # Return the updated model parameters
        return self.model.get_params()
    
    def evaluate(self, global_model=None):
        """Evaluate the model on local test data"""
        if global_model is not None:
            self.model.load_state_dict(copy.deepcopy(global_model.state_dict()))
        
        self.model.eval()
        test_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False
        )
        
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Sum up batch loss
                test_loss += self.criterion(output, target).item()
                
                # Get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = correct / len(self.test_data)
        
        return test_loss, accuracy 