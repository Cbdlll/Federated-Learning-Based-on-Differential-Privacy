#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import numpy as np
from models.lenet import LeNet

def load_model(model_path, device='cpu'):
    """Load a saved model for inference"""
    model = LeNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def predict_single_image(model, image_tensor, device):
    """Make prediction for a single image"""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image_tensor)
        prediction = output.argmax(dim=1, keepdim=True).item()
        probability = torch.nn.functional.softmax(output, dim=1)[0]
    return prediction, probability

def show_prediction(image, prediction, probability):
    """Display the image with its prediction"""
    # Convert tensor to numpy for display
    if isinstance(image, torch.Tensor):
        image = image.squeeze().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(f'Prediction: {prediction}')
    plt.axis('off')
    
    # Plot probability distribution
    plt.figure(figsize=(10, 4))
    plt.bar(range(10), probability.cpu().numpy())
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.xticks(range(10))
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Inference with saved Federated Learning model')
    parser.add_argument('--model_path', type=str, default='models/saved/federated_model.pth', 
                        help='Path to saved model')
    parser.add_argument('--num_samples', type=int, default=5, 
                        help='Number of samples to test')
    args = parser.parse_args()
    
    # Configure device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Select random samples from test set
    indices = torch.randperm(len(test_dataset))[:args.num_samples]
    
    # Make predictions
    print(f"Making predictions for {args.num_samples} random test samples...")
    
    for i in range(args.num_samples):
        idx = indices[i].item()
        image, true_label = test_dataset[idx]
        
        # Get the original image for display (before normalization)
        orig_image = test_dataset.data[idx].numpy()
        
        # Make prediction
        prediction, probability = predict_single_image(model, image, device)
        
        print(f"Sample {i+1}:")
        print(f"  True label: {true_label}")
        print(f"  Predicted: {prediction}")
        print(f"  Confidence: {probability[prediction]:.4f}")
        print()
        
        # Display result
        show_prediction(orig_image, prediction, probability)

if __name__ == "__main__":
    main() 