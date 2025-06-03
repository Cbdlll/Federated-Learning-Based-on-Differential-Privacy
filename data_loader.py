#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, Subset, random_split
from typing import List, Tuple, Dict

class MNISTDataset(Dataset):
    """MNIST wrapper for easier handling in federated learning context"""
    def __init__(self, data, targets, transform=None, device=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)
            
        # Move to device if specified and GPU is available
        if self.device == 'cuda' and torch.cuda.is_available():
            img = img.to(self.device, non_blocking=True)
            target = torch.tensor(target).to(self.device, non_blocking=True)

        return img, target

    def __len__(self):
        return len(self.data)

def load_and_split_data(num_clients: int = 10, iid: bool = True, device: str = 'cpu') -> Tuple[List[Dataset], List[Dataset]]:
    """
    Load MNIST data and split among clients
    
    Args:
        num_clients: Number of clients to split data for
        iid: If True, split data randomly (IID), else split by labels (non-IID)
        device: Device to load data on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (train_datasets, test_datasets) where each is a list of datasets for each client
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Split training and test data for clients
    train_datasets = []
    test_datasets = []
    
    if iid:
        # IID: Random split
        train_data_per_client = len(train_dataset) // num_clients
        test_data_per_client = len(test_dataset) // num_clients
        
        # Split training data
        train_datasets = random_split(
            train_dataset, 
            [train_data_per_client] * num_clients
        )
        
        # Split test data
        test_datasets = random_split(
            test_dataset,
            [test_data_per_client] * num_clients
        )
    else:
        # Non-IID: Sort by label, then split
        # Group indices by label
        train_label_indices = {}
        for idx, (_, label) in enumerate(train_dataset):
            if label not in train_label_indices:
                train_label_indices[label] = []
            train_label_indices[label].append(idx)
            
        test_label_indices = {}
        for idx, (_, label) in enumerate(test_dataset):
            if label not in test_label_indices:
                test_label_indices[label] = []
            test_label_indices[label].append(idx)
            
        # Distribute labels among clients (2 labels per client for MNIST with 10 clients)
        labels_per_client = 2
        client_labels = {}
        all_labels = list(range(10))
        np.random.shuffle(all_labels)
        
        for i in range(num_clients):
            client_labels[i] = all_labels[(i * labels_per_client) % 10 : ((i + 1) * labels_per_client) % 10 or 10]
            
        # Create datasets for each client
        for client_idx in range(num_clients):
            # Training data
            client_train_indices = []
            for label in client_labels[client_idx]:
                client_train_indices.extend(train_label_indices[label])
            
            # Test data
            client_test_indices = []
            for label in client_labels[client_idx]:
                client_test_indices.extend(test_label_indices[label])
                
            train_datasets.append(Subset(train_dataset, client_train_indices))
            test_datasets.append(Subset(test_dataset, client_test_indices))
    
    return train_datasets, test_datasets 