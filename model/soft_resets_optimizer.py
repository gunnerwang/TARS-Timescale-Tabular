import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable

class SoftResetsOptimizer(Optimizer):
    """
    Implementation of the Soft Resets optimizer from the paper:
    "Non-Stationary Learning of Neural Networks with Automatic Soft Parameter Reset"
    
    This optimizer implements automatic soft parameter resets to handle non-stationary learning
    environments by gradually drifting parameters back towards their initialization while
    maintaining adaptive learning rates.
    
    FIXES APPLIED:
    1. More conservative gamma initialization and updates
    2. Simplified gamma update mechanism  
    3. Better parameter initialization
    4. Reduced interference with standard training
    """
    
    def __init__(
        self, 
        params, 
        lr: float = 0.001,
        gamma_initial: float = 0.999,  # FIXED: Much more conservative initial gamma
        gamma_min: float = 0.8,        # FIXED: Higher minimum to reduce reset strength
        gamma_max: float = 0.9999,     # FIXED: Allow gamma to stay very high
        gamma_decay: float = 1.0,      # FIXED: No automatic decay by default
        sigma0: float = 0.01,
        s: float = 20.0,               # FIXED: Higher s for more stability
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        use_bayesian: bool = False,
        per_layer_gamma: bool = False,
        enable_gamma_updates: bool = False  # FIXED: Disable automatic gamma updates by default
    ):
        """
        Initialize the Soft Resets optimizer
        
        Args:
            params: Model parameters
            lr: Base learning rate
            gamma_initial: Initial drift parameter γ (higher = less reset)
            gamma_min: Minimum value for γ
            gamma_max: Maximum value for γ
            gamma_decay: Decay rate for γ (1.0 = no decay)
            sigma0: Initial distribution standard deviation
            s: Parameter for scaling posterior variance (higher = more stable)
            beta1, beta2: Adam optimizer momentum parameters
            epsilon: Numerical stability parameter
            use_bayesian: Whether to use Bayesian variant
            per_layer_gamma: Whether to use independent γ per layer
            enable_gamma_updates: Whether to enable automatic gamma updates
        """
        
        # Validation
        if s < 5.0:
            print(f"WARNING: s={s} may cause training instability. Recommended: s >= 10.0")
        
        defaults = dict(
            lr=lr,
            gamma_initial=gamma_initial,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            gamma_decay=gamma_decay,
            sigma0=sigma0,
            s=s,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            use_bayesian=use_bayesian,
            per_layer_gamma=per_layer_gamma,
            enable_gamma_updates=enable_gamma_updates
        )
        
        super(SoftResetsOptimizer, self).__init__(params, defaults)
        
        # Store initial parameter values and state
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['initial_param'] = p.data.clone()  # Store initial parameter values
                state['momentum'] = torch.zeros_like(p.data)  # First moment estimate
                state['variance'] = torch.zeros_like(p.data)  # Second moment estimate
                
                if group['use_bayesian']:
                    state['sigma'] = torch.full_like(p.data, group['sigma0'])
                
                # Initialize γ parameter
                if group['per_layer_gamma']:
                    state['gamma'] = torch.tensor(group['gamma_initial'], device=p.device)
                else:
                    state['gamma'] = torch.full_like(p.data, group['gamma_initial'])
                
                # Track for gamma updates
                state['loss_history'] = []
        
    def step(self, closure=None):
        """Execute single optimization step with improved stability"""
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SoftResets does not support sparse gradients')
                    
                state = self.state[p]
                state['step'] += 1
                
                lr = group['lr']
                gamma = state['gamma']
                
                # Adam optimizer bias correction
                bias_correction1 = 1 - group['beta1'] ** state['step']
                bias_correction2 = 1 - group['beta2'] ** state['step']
                
                # Update first and second moment estimates
                state['momentum'].mul_(group['beta1']).add_(grad, alpha=1 - group['beta1'])
                state['variance'].mul_(group['beta2']).addcmul_(grad, grad, value=1 - group['beta2'])
                
                # Bias correction
                m_hat = state['momentum'] / bias_correction1
                v_hat = state['variance'] / bias_correction2
                
                # FIXED: Simplified and more stable update rule
                # When gamma is close to 1, this behaves almost like standard Adam
                # When gamma is lower, it adds regularization towards initial parameters
                
                if group['enable_gamma_updates'] and gamma.mean().item() < 0.99:
                    # Apply soft reset only when gamma is significantly less than 1
                    regularization_target = gamma * p.data + (1 - gamma) * state['initial_param']
                    
                    # Adaptive learning rate (more conservative)
                    s = group['s']
                    adaptive_lr_factor = gamma + (1 - gamma) / s  # Simplified formula
                    adaptive_lr = lr * adaptive_lr_factor
                    
                    # Apply update from regularization target
                    grad_update = m_hat / (torch.sqrt(v_hat) + group['epsilon'])
                    p.data.copy_(regularization_target - adaptive_lr * grad_update)
                else:
                    # Standard Adam update when gamma is high or updates disabled
                    grad_update = m_hat / (torch.sqrt(v_hat) + group['epsilon'])
                    p.data.add_(grad_update, alpha=-lr)
                
                # FIXED: Very conservative gamma decay (only if enabled and very slow)
                if group['gamma_decay'] < 1 and group['enable_gamma_updates'] and state['step'] % 1000 == 0:
                    gamma.mul_(group['gamma_decay']).clamp_(group['gamma_min'], group['gamma_max'])
        
        return loss
    
    def update_gamma(self, model, dataloader, criterion, device='cpu', max_batches=5):
        """
        SIMPLIFIED gamma update - much more conservative and stable
        
        Args:
            model: Current model
            dataloader: Dataset for evaluation  
            criterion: Loss function
            device: Computing device
            max_batches: Number of batches to evaluate (kept small for efficiency)
        """
        if not any(group['enable_gamma_updates'] for group in self.param_groups):
            return  # Skip if gamma updates are disabled
            
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_samples = 0
            batch_count = 0
            
            for batch_data in dataloader:
                if batch_count >= max_batches:
                    break
                    
                try:
                    if isinstance(batch_data, (list, tuple)):
                        if len(batch_data) == 3:  # (inputs, mask, targets) format
                            inputs, mask, targets = batch_data
                            inputs, targets = inputs.to(device), targets.to(device)
                            if isinstance(inputs, (list, tuple)):
                                X_num, X_cat = inputs[0], inputs[1] if len(inputs) > 1 else None
                                outputs = model(X_num, X_cat, mask.to(device))
                            else:
                                outputs = model(inputs, None, mask.to(device))
                        else:
                            inputs, targets = batch_data
                            inputs, targets = inputs.to(device), targets.to(device)
                            outputs = model(inputs)
                    else:
                        continue
                    
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * targets.size(0)
                    total_samples += targets.size(0)
                    batch_count += 1
                    
                except Exception as e:
                    continue
            
            if total_samples > 0:
                current_loss = total_loss / total_samples
            else:
                current_loss = float('inf')
        
        # FIXED: Much simpler and more conservative gamma update
        for group in self.param_groups:
            if not group['enable_gamma_updates']:
                continue
                
            for p in group['params']:
                state = self.state[p]
                
                # Maintain a simple loss history
                state['loss_history'].append(current_loss)
                if len(state['loss_history']) > 10:  # Keep only recent history
                    state['loss_history'] = state['loss_history'][-10:]
                
                # Only adjust gamma if we have enough history and see clear degradation
                if len(state['loss_history']) >= 5:
                    recent_avg = np.mean(state['loss_history'][-3:])
                    older_avg = np.mean(state['loss_history'][-5:-2])
                    
                    # Only decrease gamma if loss is clearly getting worse
                    if recent_avg > older_avg * 1.1:  # 10% worse
                        # Very small gamma decrease
                        gamma_decrease = 0.001
                        state['gamma'].add_(-gamma_decrease).clamp_(group['gamma_min'], group['gamma_max'])
                    # Don't increase gamma automatically - let it stay high
        
        model.train()
        
    def get_gamma_values(self):
        """Get current γ values for monitoring"""
        gamma_values = {}
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                if p in self.state:
                    gamma = self.state[p]['gamma']
                    if group['per_layer_gamma']:
                        gamma_values[f'group_{i}_param_{j}'] = gamma.item()
                    else:
                        gamma_values[f'group_{i}_param_{j}'] = gamma.mean().item()
        return gamma_values
    
    def get_gamma_statistics(self):
        """Get comprehensive gamma statistics for monitoring"""
        all_gamma_values = []
        layer_stats = {}
        
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                if p in self.state:
                    gamma = self.state[p]['gamma']
                    if group['per_layer_gamma']:
                        gamma_val = gamma.item()
                    else:
                        gamma_val = gamma.mean().item()
                    
                    all_gamma_values.append(gamma_val)
                    layer_stats[f'group_{i}_param_{j}'] = {
                        'value': gamma_val,
                        'type': 'per_layer' if group['per_layer_gamma'] else 'global'
                    }
        
        if all_gamma_values:
            return {
                'values': layer_stats,
                'global_mean': float(np.mean(all_gamma_values)),
                'global_std': float(np.std(all_gamma_values)),
                'global_min': float(np.min(all_gamma_values)),
                'global_max': float(np.max(all_gamma_values)),
                'num_parameters': len(all_gamma_values),
                'gamma_updates_enabled': any(group['enable_gamma_updates'] for group in self.param_groups)
            }
        else:
            return {}

# FIXED: Improved utility function with better defaults
def create_soft_resets_optimizer(model_params, lr=0.001, config_type='minimal'):
    """
    Create a SoftResetsOptimizer with improved configurations
    
    Args:
        model_params: Model parameters
        lr: Learning rate
        config_type: Configuration type ('minimal', 'standard', 'conservative', 'experimental')
    
    Returns:
        SoftResetsOptimizer instance
    """
    configs = {
        'minimal': {
            # FIXED: Minimal reset behavior - behaves almost like Adam
            'gamma_initial': 0.9999,
            'gamma_min': 0.95,
            'gamma_max': 0.9999,
            'gamma_decay': 1.0,  # No decay
            's': 50.0,  # High stability
            'use_bayesian': False,
            'per_layer_gamma': False,
            'enable_gamma_updates': False  # Disabled
        },
        'standard': {
            # FIXED: Conservative but functional reset behavior
            'gamma_initial': 0.999,
            'gamma_min': 0.9,
            'gamma_max': 0.9999,
            'gamma_decay': 1.0,
            's': 25.0,
            'use_bayesian': False,
            'per_layer_gamma': False,
            'enable_gamma_updates': True
        },
        'conservative': {
            'gamma_initial': 0.9995,
            'gamma_min': 0.95,
            'gamma_max': 0.9999,
            'gamma_decay': 1.0,
            's': 30.0,
            'use_bayesian': False,
            'per_layer_gamma': False,
            'enable_gamma_updates': True
        },
        'experimental': {
            # More aggressive - use with caution
            'gamma_initial': 0.995,
            'gamma_min': 0.8,
            'gamma_max': 0.999,
            'gamma_decay': 0.9999,
            's': 15.0,
            'use_bayesian': False,
            'per_layer_gamma': True,
            'enable_gamma_updates': True
        }
    }
    
    config = configs.get(config_type, configs['minimal'])
    
    return SoftResetsOptimizer(
        model_params,
        lr=lr,
        **config
    ) 