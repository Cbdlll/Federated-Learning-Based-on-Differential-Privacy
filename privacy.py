#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from typing import List, Optional, Literal

class PrivacyEngine:
    """
    Implementation of Differential Privacy for Deep Learning using 
    Differentially Private Stochastic Gradient Descent (DP-SGD)
    
    Supports both Gaussian and Laplace noise mechanisms.
    """
    def __init__(self, 
                 model: torch.nn.Module,
                 batch_size: int,
                 sample_size: int,
                 alphas: List[float],
                 noise_multiplier: Optional[float] = None,
                 max_grad_norm: float = 1.0,
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 noise_type: Literal["gaussian", "laplace"] = "gaussian"):
        """
        Initialize the privacy engine
        
        Args:
            model: The model to be trained with differential privacy
            batch_size: The batch size being used for training
            sample_size: The size of the training dataset
            alphas: A list of RDP orders
            noise_multiplier: The noise multiplier for DP-SGD
            max_grad_norm: The maximum L2 norm of per-sample gradients
            epsilon: The privacy budget epsilon
            delta: The privacy budget delta
            noise_type: Type of noise to add ("gaussian" or "laplace")
        """
        self.model = model
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.alphas = alphas
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon
        self.delta = delta
        self.device = next(model.parameters()).device
        self.noise_type = noise_type
        
        # If noise_multiplier is not provided, estimate it from epsilon and delta
        if noise_multiplier is None:
            # This is a simplified version - in practice, we should use binary search
            # to find the optimal noise multiplier for the given epsilon and delta
            steps = 60000 // batch_size  # Assuming one epoch over MNIST
            self.noise_multiplier = 1.0 / epsilon
        else:
            self.noise_multiplier = noise_multiplier
            
        # Record privacy budget spent
        self.steps = 0
        
    def attach(self, optimizer):
        """
        Attach the privacy engine to the optimizer
        
        Args:
            optimizer: The optimizer to attach the privacy engine to
        """
        # Save original step method
        self.original_step = optimizer.step
        
        # Replace the step method with our private version
        def private_step(*args, **kwargs):
            self._clip_and_add_noise(optimizer)
            self.steps += 1
            return self.original_step(*args, **kwargs)
        
        optimizer.step = private_step
        
    def _clip_and_add_noise(self, optimizer):
        """
        Clip gradients and add noise to implement differential privacy
        
        Args:
            optimizer: The optimizer being used
        """
        # Get all parameter gradients
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    # Clip gradients
                    param.grad.data = self._clip_gradients(param.grad.data)
                    
                    # Add noise based on selected mechanism
                    if self.noise_type == "gaussian":
                        noise = torch.normal(
                            mean=0,
                            std=self.noise_multiplier * self.max_grad_norm,
                            size=param.grad.shape,
                            device=param.grad.device
                        )
                    elif self.noise_type == "laplace":
                        # Laplace noise with scale parameter b = noise_multiplier * max_grad_norm / sqrt(2)
                        # For Laplace distribution with scale b, the standard deviation is sqrt(2) * b
                        scale = self.noise_multiplier * self.max_grad_norm / np.sqrt(2)
                        uniform = torch.rand(param.grad.shape, device=param.grad.device) - 0.5
                        noise = -scale * torch.sign(uniform) * torch.log(1 - 2 * torch.abs(uniform))
                    
                    param.grad.data.add_(noise)
                    
    def _clip_gradients(self, grad):
        """
        Clip gradients by their L2 norm
        
        Args:
            grad: The gradient to clip
            
        Returns:
            The clipped gradient
        """
        # Calculate L2 norm
        grad_norm = torch.norm(grad)
        
        # Clip if necessary
        if grad_norm > self.max_grad_norm:
            grad = grad * self.max_grad_norm / grad_norm
            
        return grad
    
    def get_privacy_spent(self):
        """
        Calculate privacy budget spent so far
        
        Returns:
            Tuple of (epsilon, delta)
        """
        # This is a simplified calculation for demonstration purposes
        # In practice, a proper RDP accountant should be used
        rdp = self._compute_rdp()
        eps = self._rdp_to_dp(rdp)
        return eps, self.delta
    
    def _compute_rdp(self):
        """
        Compute RDP for the current parameters
        
        Returns:
            RDP values for each alpha
        """
        # Sampling probability (probability of a data point being selected in one step)
        q = self.batch_size / self.sample_size
        
        # Compute RDP for each alpha
        rdp = []
        for alpha in self.alphas:
            if self.noise_type == "gaussian":
                # RDP for Gaussian mechanism with sampling
                temp = q * q * alpha / (self.noise_multiplier * self.noise_multiplier)
                rdp.append(temp * self.steps)
            elif self.noise_type == "laplace":
                # RDP for Laplace mechanism with sampling
                # This is an approximation - in practice use a more accurate formula
                temp = q * q * alpha * (alpha + 1) / (2 * self.noise_multiplier * self.noise_multiplier)
                rdp.append(temp * self.steps)
            
        return rdp
    
    def _rdp_to_dp(self, rdp_values):
        """
        Convert from RDP to DP guarantees
        
        Args:
            rdp_values: List of RDP values for each alpha
            
        Returns:
            The epsilon value for DP
        """
        # Select the best alpha
        eps = float('inf')
        for rdp, alpha in zip(rdp_values, self.alphas):
            # Convert RDP to DP
            # This is a simplified version - in practice use a proper implementation
            current = rdp + (np.log(1 / self.delta) / (alpha - 1))
            eps = min(eps, current)
            
        return eps 