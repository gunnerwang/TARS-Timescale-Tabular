import math
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
from typing import Optional, Dict, Any

from model.models.ftt import MultiheadAttention, Tokenizer, get_activation_fn, get_nonglu_activation_fn
from model.lib.temporal_embeddings import TemporalEmbeddings, TimeAttentionEmbeddings, MemoryBankTimeAttentionEmbeddings
from model.cbp_linear import CBPLinear

DATASET_SIZE = 227087     # ct: 227087, de: 279415, mr: 160019, we: 340596
BATCH_SIZE = 1024


def calculate_adaptive_warmup_params(dataset_size, batch_size):
    """Calculate adaptive warm-up parameters based on dataset characteristics"""
    # Calculate updates per epoch
    updates_per_epoch = max(1, dataset_size // batch_size)
    
    # Define warm-up strategy based on dataset size category
    if dataset_size < 10000:
        # Small datasets: Conservative warm-up
        importance_start_epochs = 1.0    # Start importance after 1 epoch
        conservative_end_epochs = 3.0    # Conservative period until 3 epochs  
        temperature_annealing_epochs = 4.0  # Annealing over 4 epochs
    elif dataset_size < 100000:
        # Medium datasets: Moderate warm-up
        importance_start_epochs = 1.5    # Start importance after 1.5 epochs
        conservative_end_epochs = 4.0    # Conservative period until 4 epochs
        temperature_annealing_epochs = 6.0  # Annealing over 6 epochs
    else:
        # Large datasets: Extended warm-up for stability
        importance_start_epochs = 2.0    # Start importance after 2 epochs
        conservative_end_epochs = 4.0    # Conservative period until 4 epochs
        temperature_annealing_epochs = 6.0  # Annealing over 6 epochs
    
    # Convert epochs to update counts
    min_updates_for_importance = int(importance_start_epochs * updates_per_epoch)
    conservative_end_updates = int(conservative_end_epochs * updates_per_epoch)
    temperature_warmup_steps = int(temperature_annealing_epochs * updates_per_epoch)
    
    return {
        'updates_per_epoch': updates_per_epoch,
        'min_updates_for_importance': min_updates_for_importance,
        'conservative_end_updates': conservative_end_updates, 
        'temperature_warmup_steps': temperature_warmup_steps,
        'dataset_category': 'small' if dataset_size < 10000 else 'medium' if dataset_size < 100000 else 'large',
        'importance_start_epochs': importance_start_epochs,
        'conservative_end_epochs': conservative_end_epochs,
        'temperature_annealing_epochs': temperature_annealing_epochs
    }


class MultiTimescaleImplicitEncoder(nn.Module):
    """Multi-timescale drift detector with scale-specific indicators and feature importance.
    
    Each timescale focuses on different aspects of drift:
    - Fast: Immediate deviations and outlier detection (working memory)
    - Slow: Distributional shifts and trend changes (medium-term adaptation)  
    - Ultra: Structural changes and long-term stability (long-term plasticity)
    
    Features adaptive importance weighting to focus on drift-prone features.
    """
    def __init__(self, d_in, d_out, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE, 
                 test_time_adaptation=True, drift_significance_threshold=0.1):
        super().__init__()
        self.d_in = d_in
        self.test_time_adaptation = test_time_adaptation  # Enable test-time adaptation
        self.drift_significance_threshold = drift_significance_threshold  # Threshold for significant drift
        
        # Calculate adaptive warm-up parameters based on dataset characteristics
        self.warmup_params = calculate_adaptive_warmup_params(dataset_size, batch_size)
        
        print(f"🔧 FTT Adaptive Warm-up Configuration for {self.warmup_params['dataset_category']} dataset:")
        print(f"   ├── Dataset Size: {dataset_size:,} samples")
        print(f"   ├── Batch Size: {batch_size}")
        print(f"   ├── Updates per Epoch: {self.warmup_params['updates_per_epoch']}")
        print(f"   ├── Feature Importance Start: {self.warmup_params['importance_start_epochs']:.1f} epochs ({self.warmup_params['min_updates_for_importance']} updates)")
        print(f"   ├── Conservative Period End: {self.warmup_params['conservative_end_epochs']:.1f} epochs ({self.warmup_params['conservative_end_updates']} updates)")
        print(f"   ├── Temperature Annealing: {self.warmup_params['temperature_annealing_epochs']:.1f} epochs ({self.warmup_params['temperature_warmup_steps']} updates)")
        print(f"   ├── Test-time Adaptation: {'Enabled' if test_time_adaptation else 'Disabled'}")
        print(f"   └── Drift Significance Threshold: {drift_significance_threshold}")
        
        # Three hierarchical timescales with different focus
        self.fast_momentum = 0.9    # Working memory timescale (more responsive)
        self.slow_momentum = 0.99   # Medium-term adaptation  
        self.ultra_momentum = 0.999 # Long-term structural plasticity
        
        # Buffers for each timescale's statistics
        register_buffer = getattr(self, "register_buffer", None)
        if callable(register_buffer):
            # Fast timescale (working memory-like)
            self.register_buffer("fast_mean", torch.zeros(d_in))
            self.register_buffer("fast_var", torch.ones(d_in))
            self.register_buffer("fast_min", torch.zeros(d_in))
            self.register_buffer("fast_max", torch.ones(d_in))
            
            # Slow timescale (medium adaptation)
            self.register_buffer("slow_mean", torch.zeros(d_in))
            self.register_buffer("slow_var", torch.ones(d_in))
            self.register_buffer("slow_skew", torch.zeros(d_in))
            
            # Ultra timescale (structural plasticity)
            self.register_buffer("ultra_mean", torch.zeros(d_in))
            self.register_buffer("ultra_var", torch.ones(d_in))
            self.register_buffer("ultra_corr_sum", torch.zeros(d_in, d_in))
            
            # Feature importance weights (learned adaptively)
            self.register_buffer("feature_importance", torch.ones(d_in))
            self.register_buffer("drift_history", torch.zeros(d_in))
            self.register_buffer("update_count", torch.zeros(1))
            
            # Drift significance tracking
            self.register_buffer("drift_magnitude_history", torch.zeros(100))  # Rolling window of drift magnitudes
            self.register_buffer("drift_history_ptr", torch.zeros(1, dtype=torch.long))  # Circular buffer pointer
            self.register_buffer("drift_significance", torch.tensor(1.0))  # Current drift significance score
        else:
            # Fast timescale
            self.fast_mean = nn.Parameter(torch.zeros(d_in), requires_grad=False)
            self.fast_var = nn.Parameter(torch.ones(d_in), requires_grad=False)
            self.fast_min = nn.Parameter(torch.zeros(d_in), requires_grad=False)
            self.fast_max = nn.Parameter(torch.ones(d_in), requires_grad=False)
            
            # Slow timescale
            self.slow_mean = nn.Parameter(torch.zeros(d_in), requires_grad=False)
            self.slow_var = nn.Parameter(torch.ones(d_in), requires_grad=False)
            self.slow_skew = nn.Parameter(torch.zeros(d_in), requires_grad=False)
            
            # Ultra timescale
            self.ultra_mean = nn.Parameter(torch.zeros(d_in), requires_grad=False)
            self.ultra_var = nn.Parameter(torch.ones(d_in), requires_grad=False)
            self.ultra_corr_sum = nn.Parameter(torch.zeros(d_in, d_in), requires_grad=False)
            
            # Feature importance weights
            self.feature_importance = nn.Parameter(torch.ones(d_in), requires_grad=False)
            self.drift_history = nn.Parameter(torch.zeros(d_in), requires_grad=False)
            self.update_count = nn.Parameter(torch.zeros(1), requires_grad=False)
            
            # Drift significance tracking
            self.drift_magnitude_history = nn.Parameter(torch.zeros(100), requires_grad=False)
            self.drift_history_ptr = nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)
            self.drift_significance = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        
        # Scale-specific projections with different output dimensions
        fast_dim = d_out // 3
        slow_dim = d_out // 3
        ultra_dim = d_out - fast_dim - slow_dim
        
        # Fast scale: focus on immediate deviations (outlier detection)
        fast_features = d_in * 2  # z_scores + outlier_indicators
        self.fast_proj = nn.Linear(fast_features, fast_dim)
        
        # Slow scale: focus on distributional changes
        slow_features = d_in * 2  # kl_div + skewness_change
        self.slow_proj = nn.Linear(slow_features, slow_dim)
        
        # Ultra scale: focus on structural/correlation changes
        ultra_features = d_in + min(d_in * (d_in - 1) // 2, 64)  # mean_shift + top_correlations
        self.ultra_proj = nn.Linear(ultra_features, ultra_dim)
        
        # Feature importance adaptation
        self.importance_decay = 0.95
        self.importance_threshold = 0.1

    def _update_drift_significance(self, current_drift_magnitude):
        """Update drift significance score based on recent drift history"""
        if self.update_count > 10:  # Only after some warm-up
            # Update circular buffer with current drift magnitude
            ptr = int(self.drift_history_ptr.item())
            self.drift_magnitude_history[ptr] = current_drift_magnitude
            self.drift_history_ptr.data = torch.tensor((ptr + 1) % 100)
            
            # Calculate drift significance as average of recent drift magnitudes
            if self.update_count > 100:
                # Use full buffer
                recent_drift = self.drift_magnitude_history.mean()
            else:
                # Use only filled portion of buffer
                filled_size = int(min(self.update_count.item(), 100))
                recent_drift = self.drift_magnitude_history[:filled_size].mean()
            
            # Update drift significance with smoothing
            self.drift_significance.data = 0.9 * self.drift_significance + 0.1 * recent_drift
            
    def is_drift_significant(self):
        """Check if current drift is significant enough to warrant drift-based routing"""
        return self.drift_significance.item() > self.drift_significance_threshold
        
    def _compute_fast_indicators(self, x, batch_size):
        """Fast scale: Immediate deviations and outlier detection with adaptive importance weighting"""
        
        # --- 1. Z-score for deviation from historical mean ---
        z_scores = (x - self.fast_mean) / (torch.sqrt(self.fast_var) + 1e-5)
        z_scores = torch.clamp(z_scores, min=-6.0, max=6.0)  # Optional: stabilize spikes

        # --- 2. Outlier detection via normalized position ---
        range_width = self.fast_max - self.fast_min + 1e-5
        normalized_pos = (x - self.fast_min) / range_width
        outlier_indicators = torch.sigmoid(10 * (normalized_pos - 1)) + torch.sigmoid(-10 * normalized_pos)

        # --- 3. Concatenate drift features first ---
        fast_features = torch.cat([z_scores, outlier_indicators], dim=1)
        
        # --- 4. Apply importance weighting with warm-up gate ---
        min_updates = self.warmup_params['min_updates_for_importance']
        conservative_end = self.warmup_params['conservative_end_updates']
        
        if self.update_count.item() > min_updates:
            # Gradual warm-up: ramp up importance usage over time
            if self.update_count.item() < conservative_end:
                # Conservative importance application during early stable period
                warmup_ratio = min((self.update_count.item() - min_updates) / (conservative_end - min_updates), 1.0)
                importance = self.feature_importance * warmup_ratio + (1.0 - warmup_ratio)
            else:
                # Full importance after sufficient training
                importance = self.feature_importance
            
            # Repeat importance for both z_scores and outlier features
            importance_expanded = torch.cat([importance, importance]).unsqueeze(0)
            fast_features = fast_features * importance_expanded
        # else: no weighting during early training (keep uniform features)
        return fast_features
        
    def _compute_slow_indicators(self, x, batch_size):
        """Slow scale: Distributional shifts and trend changes with adaptive importance weighting"""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False) + 1e-5
        
        # 1. KL divergence components for distributional change
        var_ratio = self.slow_var / (batch_var + 1e-5)
        mean_diff_squared = (batch_mean - self.slow_mean).pow(2)
        mahalanobis_dist = mean_diff_squared / (self.slow_var + 1e-5)
        kl_div_components = 0.5 * (torch.log(var_ratio + 1e-5) + 
                                   (batch_var / (self.slow_var + 1e-5)) + 
                                   mahalanobis_dist - 1)
        
        # 2. Skewness change detection (third moment)
        batch_centered = x - batch_mean.unsqueeze(0)
        batch_skew = (batch_centered.pow(3).mean(dim=0) / 
                     (batch_var.pow(1.5) + 1e-5))
        skewness_change = torch.abs(batch_skew - self.slow_skew)
        
        # Concatenate drift features first
        slow_features = torch.cat([
            kl_div_components.unsqueeze(0).expand(batch_size, -1),
            skewness_change.unsqueeze(0).expand(batch_size, -1)
        ], dim=1)
        
        # Apply importance weighting with warm-up gate
        min_updates = self.warmup_params['min_updates_for_importance']
        conservative_end = self.warmup_params['conservative_end_updates']
        
        if self.update_count.item() > min_updates:
            # Gradual warm-up: ramp up importance usage over time
            if self.update_count.item() < conservative_end:
                # Conservative importance application during early stable period
                warmup_ratio = min((self.update_count.item() - min_updates) / (conservative_end - min_updates), 1.0)
                importance = self.feature_importance * warmup_ratio + (1.0 - warmup_ratio)
            else:
                # Full importance after sufficient training
                importance = self.feature_importance
            
            # Repeat importance for both kl_div and skewness features
            importance_expanded = torch.cat([importance, importance]).unsqueeze(0)
            slow_features = slow_features * importance_expanded
        # else: no weighting during early training (keep uniform features)
        return slow_features
        
    def _compute_ultra_indicators(self, x, batch_size):
        """Ultra scale: Structural changes and long-term stability with adaptive importance weighting"""
        batch_mean = x.mean(dim=0)
        
        # 1. Long-term mean shift (structural change)
        mean_shift = torch.abs(batch_mean - self.ultra_mean) / (torch.sqrt(self.ultra_var) + 1e-5)
        
        # 2. Top correlation changes (limited to prevent explosion)
        max_corr_features = min(self.d_in * (self.d_in - 1) // 2, 64)
        
        if batch_size > 1:
            # Compute batch correlation matrix
            x_centered = x - batch_mean.unsqueeze(0)
            batch_corr = torch.mm(x_centered.t(), x_centered) / (batch_size - 1)
            
            # Get reference correlation from ultra statistics
            if self.update_count > 10:  # Only after sufficient updates
                ref_corr = self.ultra_corr_sum / (self.update_count + 1e-5)
                corr_diff = torch.abs(batch_corr - ref_corr)
                
                # Extract top correlation changes (upper triangle, excluding diagonal)
                triu_indices = torch.triu_indices(self.d_in, self.d_in, offset=1)
                corr_changes = corr_diff[triu_indices[0], triu_indices[1]]
                
                # Select top changes (most significant correlation shifts)
                if len(corr_changes) > max_corr_features:
                    top_changes, _ = torch.topk(corr_changes, max_corr_features)
                    correlation_features = top_changes
                else:
                    correlation_features = corr_changes
            else:
                correlation_features = torch.zeros(max_corr_features, device=x.device)
        else:
            correlation_features = torch.zeros(max_corr_features, device=x.device)
        
        # Concatenate drift features first
        mean_shift_batch = mean_shift.unsqueeze(0).expand(batch_size, -1)
        corr_batch = correlation_features.unsqueeze(0).expand(batch_size, -1)
        ultra_features = torch.cat([mean_shift_batch, corr_batch], dim=1)
        
        # Apply importance weighting with warm-up gate (only to mean shift features)
        min_updates = self.warmup_params['min_updates_for_importance']
        conservative_end = self.warmup_params['conservative_end_updates']
        
        if self.update_count.item() > min_updates:
            # Gradual warm-up: ramp up importance usage over time
            if self.update_count.item() < conservative_end:
                # Conservative importance application during early stable period
                warmup_ratio = min((self.update_count.item() - min_updates) / (conservative_end - min_updates), 1.0)
                importance = self.feature_importance * warmup_ratio + (1.0 - warmup_ratio)
            else:
                # Full importance after sufficient training
                importance = self.feature_importance
            
            # Apply importance only to mean_shift features, keep correlation features unchanged
            importance_for_mean = importance.unsqueeze(0).expand(batch_size, -1)
            ultra_features[:, :self.d_in] = ultra_features[:, :self.d_in] * importance_for_mean
        # else: no weighting during early training (keep uniform features)
        return ultra_features
        
    def _update_feature_importance(self, drift_indicators):
        """Adaptively update feature importance based on drift history with stability checks"""
        # Use adaptive parameters based on dataset characteristics
        min_updates_for_importance = self.warmup_params['min_updates_for_importance']
        conservative_end_updates = self.warmup_params['conservative_end_updates']
        
        if (self.update_count > min_updates_for_importance and 
            drift_indicators is not None):
            
            # Extract per-feature drift signals from the structured drift indicators
            # drift_indicators contains: fast (z_scores + outliers), slow (kl + skew), ultra (mean_shift + corr)
            fast_features = drift_indicators['fast']  # [batch_size, d_in * 2]
            slow_features = drift_indicators['slow']  # [batch_size, d_in * 2] 
            
            # Extract z-scores and KL divergence components as main drift signals
            batch_size = fast_features.shape[0]
            z_scores = fast_features[:, :self.d_in].abs().mean(dim=0)  # Mean absolute z-scores across batch
            kl_components = slow_features[:, :self.d_in].abs().mean(dim=0)  # Mean KL components across batch
            
            # Combined drift signal (focus on immediate and distributional changes)
            current_drift = z_scores + kl_components
            
            # Apply noise filtering: only update if drift signal is above noise threshold
            noise_threshold = 0.1  # Ignore very small drift signals as noise
            significant_drift = torch.where(current_drift > noise_threshold, 
                                          current_drift, 
                                          torch.zeros_like(current_drift))
            
            # Conservative update rate in early stages (gradual ramp-up)
            if self.update_count < conservative_end_updates:
                # Use more conservative decay in early stages
                conservative_decay = 0.98  # Slower adaptation when data is still limited
                effective_decay = conservative_decay
            else:
                effective_decay = self.importance_decay
            
            # Update drift history with exponential moving average
            self.drift_history.data = (effective_decay * self.drift_history + 
                                     (1 - effective_decay) * significant_drift)
            
            # Only update importance if drift history shows meaningful patterns
            if self.drift_history.max() > 0.05:  # At least some features show consistent drift
                # Update feature importance (higher for features with more drift history)
                raw_importance = 1.0 + self.drift_history
                self.feature_importance.data = raw_importance / (raw_importance.sum() + 1e-5) * self.d_in
                
                # Apply threshold to avoid over-focusing
                self.feature_importance.data = torch.clamp(self.feature_importance, 
                                                         min=self.importance_threshold, 
                                                         max=5.0)
        # else: Keep uniform importance (initialized as ones) during early training
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Determine update policy based on training mode and test-time adaptation setting
        should_update_stats = self.training or (self.test_time_adaptation and batch_size > 1)
        
        if should_update_stats:
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False) + 1e-5
                
                if self.update_count > 0:
                    # Adaptive momentum based on training vs test mode
                    if self.training:
                        # Training mode: use normal momentum values
                        fast_momentum = self.fast_momentum      # 0.9
                        slow_momentum = self.slow_momentum      # 0.99  
                        ultra_momentum = self.ultra_momentum    # 0.999
                    else:
                        # Test mode: use more conservative momentum for adaptation
                        fast_momentum = 0.95   # More conservative than training (was 0.9)
                        slow_momentum = 0.995  # More conservative than training (was 0.99)
                        ultra_momentum = 0.9995 # More conservative than training (was 0.999)
                    
                    # Update fast timescale statistics (always update for drift detection)
                    self.fast_mean.data = fast_momentum * self.fast_mean + (1 - fast_momentum) * batch_mean
                    self.fast_var.data = fast_momentum * self.fast_var + (1 - fast_momentum) * batch_var
                    batch_min = x.min(dim=0)[0]
                    batch_max = x.max(dim=0)[0]
                    self.fast_min.data = torch.minimum(fast_momentum * self.fast_min + (1 - fast_momentum) * batch_min, self.fast_min)
                    self.fast_max.data = torch.maximum(fast_momentum * self.fast_max + (1 - fast_momentum) * batch_max, self.fast_max)
                    
                    # Update slow timescale statistics (conservative in test mode)
                    self.slow_mean.data = slow_momentum * self.slow_mean + (1 - slow_momentum) * batch_mean
                    self.slow_var.data = slow_momentum * self.slow_var + (1 - slow_momentum) * batch_var
                    # Update skewness
                    batch_centered = x - batch_mean.unsqueeze(0)
                    batch_skew = (batch_centered.pow(3).mean(dim=0) / (batch_var.pow(1.5) + 1e-5))
                    self.slow_skew.data = slow_momentum * self.slow_skew + (1 - slow_momentum) * batch_skew
                    
                    # Update ultra timescale statistics (very conservative in test mode)
                    if self.training:
                        # Full ultra updates only during training
                        self.ultra_mean.data = ultra_momentum * self.ultra_mean + (1 - ultra_momentum) * batch_mean
                        self.ultra_var.data = ultra_momentum * self.ultra_var + (1 - ultra_momentum) * batch_var
                        # Update correlation sum
                        x_centered = x - batch_mean.unsqueeze(0)
                        batch_corr = torch.mm(x_centered.t(), x_centered) / (batch_size - 1)
                        self.ultra_corr_sum.data = ultra_momentum * self.ultra_corr_sum + (1 - ultra_momentum) * batch_corr
                    else:
                        # Minimal ultra updates during test time (only mean/var, skip correlation)
                        self.ultra_mean.data = ultra_momentum * self.ultra_mean + (1 - ultra_momentum) * batch_mean
                        self.ultra_var.data = ultra_momentum * self.ultra_var + (1 - ultra_momentum) * batch_var
                        # Skip correlation updates in test mode for stability
                    
                else:
                    # Initialize all statistics (only during very first forward pass)
                    self.fast_mean.data = batch_mean
                    self.fast_var.data = batch_var
                    self.fast_min.data = x.min(dim=0)[0]
                    self.fast_max.data = x.max(dim=0)[0]
                    
                    self.slow_mean.data = batch_mean
                    self.slow_var.data = batch_var
                    batch_centered = x - batch_mean.unsqueeze(0)
                    self.slow_skew.data = (batch_centered.pow(3).mean(dim=0) / (batch_var.pow(1.5) + 1e-5))
                    
                    self.ultra_mean.data = batch_mean
                    self.ultra_var.data = batch_var
                    x_centered = x - batch_mean.unsqueeze(0)
                    self.ultra_corr_sum.data = torch.mm(x_centered.t(), x_centered) / (batch_size - 1)
                
                self.update_count += 1
        
        # Compute scale-specific drift indicators
        fast_features = self._compute_fast_indicators(x, batch_size)
        slow_features = self._compute_slow_indicators(x, batch_size)
        ultra_features = self._compute_ultra_indicators(x, batch_size)
        
        # Calculate overall drift magnitude for significance assessment
        with torch.no_grad():
            if should_update_stats and self.update_count > 5:
                # Compute drift magnitude as combination of all timescale indicators
                fast_magnitude = fast_features.abs().mean()
                slow_magnitude = slow_features.abs().mean()
                ultra_magnitude = ultra_features.abs().mean()
                current_drift_magnitude = (fast_magnitude + slow_magnitude + ultra_magnitude) / 3.0
                self._update_drift_significance(current_drift_magnitude)
        
        # Update feature importance using computed drift indicators (ONLY during training)
        # Feature importance should not adapt during test time to maintain model stability
        if self.training and batch_size > 1:
            with torch.no_grad():
                drift_indicators_dict = {
                    'fast': fast_features,
                    'slow': slow_features,
                    'ultra': ultra_features
                }
                self._update_feature_importance(drift_indicators_dict)
        
        # Project each timescale's indicators
        fast_output = self.fast_proj(fast_features)
        slow_output = self.slow_proj(slow_features)  
        ultra_output = self.ultra_proj(ultra_features)
        
        # Return structured multi-timescale drift encoding
        return {
            'fast': fast_output,
            'slow': slow_output,
            'ultra': ultra_output,
            'concatenated': torch.cat([fast_output, slow_output, ultra_output], dim=-1),
            'feature_importance': self.feature_importance,  # For debugging/analysis
            'drift_significant': self.is_drift_significant(),  # Drift significance indicator
            'drift_magnitude': self.drift_significance.item()  # Current drift score
        }


class TemporalComponentRouter(nn.Module):
    """Router for TimeAttentionEmbeddings components (decay, period, context).
    
    Routes between three components based on drift signals:
    - decay component (fast pathway) 
    - period component (slow pathway)
    - context component (ultra pathway)
    
    Implements residual routing mechanism for improved stability:
    α̃ = γ · α + (1 - γ) · u, where u = [1/3, 1/3, 1/3]
    
    Features adaptive bypass mode: when drift is not significant, 
    relies on temporal embeddings' self-regulation instead of drift-based routing.
    """
    def __init__(self, fast_dim, slow_dim, ultra_dim, fusion_strategy='softmax', 
                 residual_gamma=0.8, learnable_gamma=True, 
                 initial_temperature=5.0, final_temperature=1.0, temperature_schedule='linear',
                 dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE,
                 enable_bypass=True, bypass_smoothing=0.9):
        super().__init__()
        self.fusion_strategy = fusion_strategy
        self.learnable_gamma = learnable_gamma
        self.enable_bypass = enable_bypass  # Enable bypass mode for non-significant drift
        self.bypass_smoothing = bypass_smoothing  # Smoothing factor for bypass transition
        
        # Temperature parameters for stable routing
        self.initial_temperature = initial_temperature  # τ_start = 5.0 (smooth routing)
        self.final_temperature = final_temperature      # τ_end = 1.0 (sharp routing)
        self.temperature_schedule = temperature_schedule # 'linear', 'cosine', or 'fixed'
        
        # Current temperature (will be updated during training)
        self.register_buffer('current_temperature', torch.tensor(initial_temperature))
        self.register_buffer('training_step', torch.tensor(0))
        
        # Calculate adaptive temperature annealing parameters
        warmup_params = calculate_adaptive_warmup_params(dataset_size, batch_size)
        self.temperature_warmup_steps = warmup_params['temperature_warmup_steps']
        
        print(f"🌡️  FTT Temperature Annealing: {warmup_params['temperature_annealing_epochs']:.1f} epochs ({self.temperature_warmup_steps} steps)")
        if enable_bypass:
            print(f"🔄 FTT Bypass Mode: Enabled (smoothing factor: {bypass_smoothing})")
            print(f"   └── When drift is not significant, temporal embeddings handle routing autonomously")
        
        # Residual routing parameters
        if learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(residual_gamma))
        else:
            self.register_buffer('gamma', torch.tensor(residual_gamma))
        
        # Uniform reference distribution for residual routing
        self.register_buffer('uniform_weights', torch.tensor([1/3, 1/3, 1/3]))
        
        # Separate routing networks for each component with their specific dimensions
        self.decay_router = nn.Linear(fast_dim, 1)
        self.period_router = nn.Linear(slow_dim, 1) 
        self.context_router = nn.Linear(ultra_dim, 1)
        
        # Track bypass usage statistics (always initialize, regardless of enable_bypass)
        self.register_buffer('bypass_usage_count', torch.tensor(0))
        self.register_buffer('drift_routing_count', torch.tensor(0))
        
        # Self-regulation router for when bypass is active
        # This learns to weight components based on temporal patterns alone
        if enable_bypass:
            # Learnable self-regulation weights (start with uniform distribution)
            self.register_buffer('bypass_weights', torch.tensor([1/3, 1/3, 1/3]))
            self.register_buffer('bypass_adaptation_rate', torch.tensor(0.01))  # Slow adaptation rate

    def _update_bypass_weights(self, component_embeddings):
        """Adaptively update bypass weights based on temporal component effectiveness"""
        if not self.enable_bypass:
            return
            
        # Simple adaptive strategy: slightly favor components with higher activation magnitudes
        # This allows temporal embeddings to naturally express their preferences
        decay_emb, period_emb, context_emb = component_embeddings
        
        with torch.no_grad():
            # Calculate activation magnitudes for each component
            decay_mag = decay_emb.abs().mean()
            period_mag = period_emb.abs().mean()
            context_mag = context_emb.abs().mean()
            
            # Normalize to get preference weights
            total_mag = decay_mag + period_mag + context_mag + 1e-8
            target_weights = torch.tensor([
                decay_mag / total_mag,
                period_mag / total_mag, 
                context_mag / total_mag
            ], device=self.bypass_weights.device)
            
            # Slowly adapt bypass weights towards target preferences
            self.bypass_weights.data = (
                (1 - self.bypass_adaptation_rate) * self.bypass_weights + 
                self.bypass_adaptation_rate * target_weights
            )

    def _update_temperature(self):
        """Update temperature according to schedule during training"""
        if self.training and self.temperature_schedule != 'fixed':
            # Increment training step
            self.training_step += 1
            
            if self.training_step <= self.temperature_warmup_steps:
                # Compute annealing progress
                progress = self.training_step.float() / self.temperature_warmup_steps
                
                if self.temperature_schedule == 'linear':
                    # Linear annealing: τ(t) = τ_start - (τ_start - τ_end) * progress
                    self.current_temperature.data = (
                        self.initial_temperature - 
                        (self.initial_temperature - self.final_temperature) * progress
                    )
                elif self.temperature_schedule == 'cosine':
                    # Cosine annealing: τ(t) = τ_end + 0.5 * (τ_start - τ_end) * (1 + cos(π * progress))
                    self.current_temperature.data = (
                        self.final_temperature + 
                        0.5 * (self.initial_temperature - self.final_temperature) * 
                        (1 + torch.cos(torch.tensor(torch.pi) * progress))
                    )
            else:
                # After warmup, keep final temperature
                self.current_temperature.data = torch.tensor(self.final_temperature)
    
    def get_temperature_info(self):
        """Get current temperature scheduling information for monitoring/debugging"""
        return {
            'current_temperature': self.current_temperature.item(),
            'training_step': self.training_step.item(),
            'warmup_progress': min(self.training_step.item() / self.temperature_warmup_steps, 1.0),
            'schedule': self.temperature_schedule,
            'initial_temp': self.initial_temperature,
            'final_temp': self.final_temperature
        }
        
    def forward(self, drift_encoding_dict, component_embeddings):
        """
        Args:
            drift_encoding_dict: Dictionary with 'fast', 'slow', 'ultra' drift signals
                                 and 'drift_significant' boolean indicator
            component_embeddings: Tuple of (decay_emb, period_emb, context_emb)
        
        Returns:
            Weighted combination of component embeddings with adaptive bypass routing
        """
        # Update temperature schedule
        self._update_temperature()
        
        # Check if drift is significant enough to warrant drift-based routing
        drift_significant = drift_encoding_dict.get('drift_significant', True)
        
        if self.enable_bypass and not drift_significant:
            # Bypass mode: Use temporal embeddings' self-regulation
            with torch.no_grad():
                self.bypass_usage_count += 1
                
            # Update bypass weights based on component effectiveness
            self._update_bypass_weights(component_embeddings)
            
            # Use learned bypass weights (with smoothing for stability)
            component_weights = self.bypass_weights.unsqueeze(0).expand(
                component_embeddings[0].shape[0], 3
            )
            
            # Optional: Add small amount of noise to prevent over-deterministic routing
            if self.training:
                noise = torch.randn_like(component_weights) * 0.01
                component_weights = F.softmax(component_weights + noise, dim=-1)
            else:
                component_weights = F.softmax(component_weights, dim=-1)
                
        else:
            # Standard drift-based routing
            with torch.no_grad():
                self.drift_routing_count += 1
                
            # Extract drift signals for each component pathway
            fast_drift = drift_encoding_dict['fast']    # Maps to decay component
            slow_drift = drift_encoding_dict['slow']    # Maps to period component  
            ultra_drift = drift_encoding_dict['ultra']  # Maps to context component
            
            # Compute component-specific routing weights
            alpha_decay = self.decay_router(fast_drift)      # [batch_size, 1]
            alpha_period = self.period_router(slow_drift)    # [batch_size, 1]
            alpha_context = self.context_router(ultra_drift) # [batch_size, 1]
            
            # Combine routing weights with temperature-controlled softmax
            combined_alphas = torch.cat([alpha_decay, alpha_period, alpha_context], dim=-1)
            
            if self.fusion_strategy == 'softmax':
                # Apply temperature scaling: α = softmax(logits / τ)
                # Higher τ → smoother distribution (less noisy routing)
                # Lower τ → sharper distribution (more decisive routing)
                raw_weights = F.softmax(combined_alphas / self.current_temperature, dim=-1)  # [batch_size, 3]
            elif self.fusion_strategy == 'normalized':
                # For normalized strategy, apply temperature to logits before normalization
                temperature_scaled = combined_alphas / self.current_temperature
                raw_weights = temperature_scaled / (temperature_scaled.sum(dim=-1, keepdim=True) + 1e-8)
            else:
                # Direct sigmoid outputs (independent activations) - no temperature needed
                raw_weights = combined_alphas
            
            # Apply residual routing mechanism: α̃ = γ · α + (1 - γ) · u
            # where u = [1/3, 1/3, 1/3] is the uniform reference distribution
            batch_size = raw_weights.shape[0]
            uniform_weights = torch.tensor([1/3, 1/3, 1/3], device=raw_weights.device)
            uniform_batch = uniform_weights.unsqueeze(0).expand(batch_size, 3)  # [batch_size, 3]
            component_weights = self.gamma * raw_weights + (1.0 - self.gamma) * uniform_batch
        
        # Weighted combination of temporal components
        decay_emb, period_emb, context_emb = component_embeddings
        
        weighted_embedding = (
            component_weights[:, 0:1] * decay_emb +    # Fast drift → Decay component
            component_weights[:, 1:2] * period_emb +   # Slow drift → Period component  
            component_weights[:, 2:3] * context_emb    # Ultra drift → Context component
        )
        
        return weighted_embedding
    
    def get_routing_info(self):
        """Get routing statistics for monitoring/debugging"""
        total_usage = self.bypass_usage_count + self.drift_routing_count + 1e-8
        return {
            'bypass_usage_ratio': self.bypass_usage_count.item() / total_usage,
            'drift_routing_ratio': self.drift_routing_count.item() / total_usage,
            'bypass_weights': self.bypass_weights.tolist() if self.enable_bypass else None,
            'current_temperature': self.current_temperature.item(),
            'total_routing_calls': int(total_usage)
        }


class CBPMultiheadAttention(nn.Module):
    """CBP-enabled MultiheadAttention for Transformer with continual neuron replacement"""
    def __init__(self, original_attention, replacement_rate=1e-4, maturity_threshold=100, cbp_init='kaiming'):
        super().__init__()
        self.original_attention = original_attention
        
        # CBP layer for the linear projections in attention
        # MultiheadAttention typically has in_proj and out_proj
        if hasattr(original_attention, 'in_proj') and hasattr(original_attention, 'out_proj'):
            self.cbp_layer = CBPLinear(
                in_layer=original_attention.in_proj,
                out_layer=original_attention.out_proj,
                replacement_rate=replacement_rate,
                maturity_threshold=maturity_threshold,
                init=cbp_init,
                act_type='identity'  # Attention doesn't use activation between in_proj and out_proj
            )
        else:
            self.cbp_layer = None
            print("Warning: CBP not applied to attention - linear layers not found")
    
    def forward(self, query, key, key_compression=None, value_compression=None):
        # Forward pass through original attention
        output = self.original_attention(query, key, key_compression, value_compression)
        
        # Apply CBP monitoring if available
        if self.cbp_layer is not None and self.training:
            # CBP monitoring (doesn't change the output, just monitors neuron utility)
            _ = self.cbp_layer(output)
        
        return output


class Transformer_Temporal(nn.Module):
    
    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        # temporal embeddings
        t_mean: float,
        t_std: float,
        temporal_embeddings: ty.Optional[Dict[str, Any]],
        # temporal enhancement parameters
        implicit_time_dim: int = 64,  # Dimension of implicit time encoding, 0 means not using it
        # Test-time adaptation parameters
        test_time_adaptation: bool = True,  # Enable adaptive statistics update during inference
        # Drift significance parameters
        drift_significance_threshold: float = 0.1,  # Threshold for significant drift detection
        enable_bypass: bool = True,  # Enable bypass routing for non-significant drift
        # CBP parameters
        use_cbp: bool = False,
        replacement_rate: float = 1e-4,
        maturity_threshold: int = 100,
        cbp_init: str = 'kaiming',
        #
        d_out: int,
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        
        self.d_numerical = d_numerical
        self.implicit_time_dim = implicit_time_dim
        self.use_cbp = use_cbp
        self.test_time_adaptation = test_time_adaptation
        self.drift_significance_threshold = drift_significance_threshold
        self.enable_bypass = enable_bypass
        
        # Initialize multi-timescale implicit encoder
        if self.implicit_time_dim > 0:
            self.implicit_time_encoder = MultiTimescaleImplicitEncoder(
                d_in=d_numerical, 
                d_out=implicit_time_dim,
                dataset_size=DATASET_SIZE,
                batch_size=BATCH_SIZE,
                test_time_adaptation=test_time_adaptation,
                drift_significance_threshold=drift_significance_threshold
            )
            
        # Create unified temporal embeddings 
        # TimeAttentionEmbeddings already contains multi-timescale components:
        # - decay: captures fast/recent changes
        # - period: captures medium-term periodic patterns  
        # - context: captures long-term structural patterns
        if temporal_embeddings is not None:
            base_config = temporal_embeddings.copy()
            
            # Single unified temporal embedding with multi-timescale capability
            unified_config = base_config.copy()
            unified_config.update({
                'decay_factor': base_config.get('decay_factor', 0.1),
                'periodic_patterns': base_config.get('periodic_patterns', [1, 24, 24*7]),  # Hour, daily, weekly patterns
                'd_embedding': base_config.get('d_embedding', 64),
                'feature_fusion': base_config.get('feature_fusion', True)
            })
            self.temporal_embeddings = MemoryBankTimeAttentionEmbeddings(t_mean, t_std, **unified_config)
            
            temporal_dim = self.temporal_embeddings.out_dim
            
            # Initialize router to combine temporal components based on drift signals
            if self.implicit_time_dim > 0:
                # Calculate actual drift dimensions (matching MultiTimescaleImplicitEncoder)
                fast_dim = implicit_time_dim // 3
                slow_dim = implicit_time_dim // 3
                ultra_dim = implicit_time_dim - fast_dim - slow_dim  # Handle remainder
                
                self.temporal_router = TemporalComponentRouter(
                    fast_dim=fast_dim,
                    slow_dim=slow_dim, 
                    ultra_dim=ultra_dim,
                    fusion_strategy='softmax',
                    initial_temperature=5.0,  # Start with smooth routing (reduce noise)
                    final_temperature=1.0,    # End with sharp routing (decisive)
                    temperature_schedule='linear',  # Gradual annealing
                    dataset_size=DATASET_SIZE,
                    batch_size=BATCH_SIZE,
                    enable_bypass=enable_bypass
                )
            
        else:
            temporal_dim = 0
            
        # Calculate temporal dimension for tokenizer
        if hasattr(self, 'temporal_embeddings'):
            if self.implicit_time_dim > 0 and hasattr(self, 'temporal_router'):
                # Router mode: use component dimension
                temporal_contribution = self.temporal_embeddings.component_dim
            else:
                # Fallback mode: use full temporal embedding dimension
                temporal_contribution = temporal_dim
        else:
            temporal_contribution = 0
            
        # Update d_numerical for tokenizer (include temporal features)
        self.tokenizer_d_numerical = self.d_numerical + temporal_contribution
        
        # Add fusion gate for adaptive temporal-numerical feature interaction
        if temporal_contribution > 0:
            self.fusion_gate = nn.Linear(temporal_contribution, self.d_numerical)
        
        # FTT-specific tokenizer (preserve original FTT architecture)
        self.tokenizer = Tokenizer(self.tokenizer_d_numerical, categories, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        
        # Initialize CBP layers list if CBP is enabled
        self.cbp_layers = nn.ModuleList() if use_cbp else None
        
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)
            
            # Add CBP layer for this transformer layer if enabled
            if use_cbp:
                # CBP for feed-forward network (linear0 -> linear1)
                cbp_layer = CBPLinear(
                    in_layer=layer['linear0'],
                    out_layer=layer['linear1'],
                    replacement_rate=replacement_rate,
                    maturity_threshold=maturity_threshold,
                    init=cbp_init,
                    act_type='relu' if not activation.endswith('glu') else 'gelu'
                )
                self.cbp_layers.append(cbp_layer)
                print(f"      CBP Layer {layer_idx+1}: {d_token} → {d_hidden} → {d_token} (FFN)")

        self.activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)
        
        # CBP for head layer if enabled
        if use_cbp and n_layers > 0:
            # Find the last layer's linear1 for CBP connection to head
            last_layer = self.layers[-1]
            cbp_head = CBPLinear(
                in_layer=last_layer['linear1'],
                out_layer=self.head,
                replacement_rate=replacement_rate,
                maturity_threshold=maturity_threshold,
                init=cbp_init,
                act_type='identity'  # No activation before head
            )
            self.cbp_layers.append(cbp_head)
            print(f"      CBP Head: {d_token} → {d_out} (Head)")
        
        # Print CBP configuration
        if self.use_cbp:
            print(f"🔄 FTT CBP (Continual Backpropagation) ENABLED")
            print(f"   ├── Replacement Rate: {replacement_rate:.2e} ({replacement_rate*100:.4f}%)")
            print(f"   ├── Maturity Threshold: {maturity_threshold} steps")
            print(f"   ├── Initialization Method: {cbp_init}")
            print(f"   └── Total CBP Layers: {len(self.cbp_layers) if self.cbp_layers else 0}")
        else:
            print("📋 Using standard FTT Transformer (CBP disabled)")
        
        # Print configuration summary
        print(f"\n🎯 Temporal-FTT Configuration Summary:")
        print(f"   ├── Numerical Features: {d_numerical}")
        if hasattr(self, 'temporal_embeddings'):
            if self.implicit_time_dim > 0 and hasattr(self, 'temporal_router'):
                temporal_info = f"{self.temporal_embeddings.component_dim} (router mode)"
            else:
                temporal_info = f"{temporal_dim} (direct mode)"
            print(f"   ├── Temporal Features: {temporal_info}")
            if hasattr(self, 'fusion_gate'):
                print(f"   ├── Fusion Gate: Enabled (adaptive temporal-numerical interaction)")
            else:
                print(f"   ├── Fusion Gate: Disabled (direct concatenation)")
        else:
            print(f"   ├── No Temporal Features")
        print(f"   ├── Tokenizer Input: {self.tokenizer_d_numerical}")
        print(f"   ├── FTT Tokens: {n_tokens}")
        print(f"   ├── Transformer Layers: {n_layers}")
        print(f"   ├── Token Dimension: {d_token}")
        print(f"   ├── Attention Heads: {n_heads}")
        print(f"   ├── KV Compression: {kv_compression if kv_compression else 'Disabled'}")
        print(f"   ├── Drift Significance Threshold: {drift_significance_threshold}")
        print(f"   ├── Bypass Routing: {'Enabled' if enable_bypass else 'Disabled'}")
        print(f"   ├── Test-time Adaptation: {'Enabled' if test_time_adaptation else 'Disabled'}")
        print(f"   └── Implicit Time Encoding: {implicit_time_dim} dims" if implicit_time_dim > 0 else "   └── Implicit Time Encoding: Disabled")

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor], idx) -> Tensor:
        # Generate multi-timescale drift encoding (structured output) based on numerical features
        if self.implicit_time_dim > 0 and x_num is not None:
            drift_encoding_dict = self.implicit_time_encoder(x_num)
            drift_significant = drift_encoding_dict.get('drift_significant', True)
            drift_magnitude = drift_encoding_dict.get('drift_magnitude', 0.0)
        else:
            drift_encoding_dict = None
            drift_significant = True  # Default to drift-based routing if no encoder
            drift_magnitude = 0.0
        
        # Generate temporal embeddings using component routing
        if hasattr(self, 'temporal_embeddings'):
            # Use router if available with drift signals
            if hasattr(self, 'temporal_router') and self.implicit_time_dim > 0 and drift_encoding_dict is not None:
                # Get separate temporal components
                component_embeddings = self.temporal_embeddings.forward_components(idx)
                
                # Use router to combine components based on drift signals (with bypass capability)
                temporal_emb = self.temporal_router(drift_encoding_dict, component_embeddings)
                
                # Optional: Log routing information during training for monitoring
                if self.training:
                    if not hasattr(self, '_routing_log_step'):
                        self._routing_log_step = 0
                    self._routing_log_step += 1
                    if self._routing_log_step % 1000 == 0:  # Log every 1000 steps
                        routing_info = self.temporal_router.get_routing_info()
                        print(f"📊 FTT Routing Stats (Step {self._routing_log_step}):")
                        print(f"   ├── Bypass Usage: {routing_info['bypass_usage_ratio']:.2%}")
                        print(f"   ├── Drift Routing: {routing_info['drift_routing_ratio']:.2%}")
                        print(f"   ├── Current Drift Magnitude: {drift_magnitude:.4f}")
                        print(f"   ├── Drift Significant: {drift_significant}")
                        if routing_info['bypass_weights']:
                            print(f"   └── Bypass Weights: {[f'{w:.3f}' for w in routing_info['bypass_weights']]}")
                    
            else:
                # Fall back to unified temporal embeddings
                temporal_emb = self.temporal_embeddings(idx)
                
            # Apply adaptive fusion with numerical features if fusion gate exists
            if hasattr(self, 'fusion_gate') and x_num is not None:
                temporal_weights = torch.sigmoid(self.fusion_gate(temporal_emb))
                x_num_weighted = x_num * temporal_weights
                x_num_fused = x_num_weighted + x_num * 0.2  # Residual connection
                # Concatenate temporal embeddings with fused numerical features
                x_num = torch.cat([x_num_fused, temporal_emb], dim=-1)
            else:
                # Fallback: direct concatenation
                if x_num is not None:
                    x_num = torch.cat([x_num, temporal_emb], dim=-1)
        else:
            temporal_emb = None
            
        # FTT-specific tokenization (preserve original FTT architecture)
        x = self.tokenizer(x_num, x_cat)

        # FTT-specific transformer layers with optional CBP
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            # Attention block
            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            # Feed-forward block with optional CBP
            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            
            # Apply CBP monitoring if enabled
            if self.use_cbp and self.cbp_layers and layer_idx < len(self.cbp_layers) - (1 if hasattr(self, 'head') else 0):
                x_residual = self.cbp_layers[layer_idx](x_residual)
                
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        # FTT-specific head processing
        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        
        # Apply head with optional CBP
        x = self.head(x)
        if self.use_cbp and self.cbp_layers and len(self.cbp_layers) > len(self.layers):
            # Apply final CBP layer for head
            x = self.cbp_layers[-1](x)
            
        x = x.squeeze(-1)
        return x
    
    def set_test_time_adaptation(self, enable: bool):
        """Enable or disable test-time adaptation of statistics
        
        Args:
            enable: If True, statistics will adapt during inference.
                   If False, statistics remain frozen after training.
        """
        self.test_time_adaptation = enable
        if hasattr(self, 'implicit_time_encoder'):
            self.implicit_time_encoder.test_time_adaptation = enable
        
        mode_str = "ENABLED" if enable else "DISABLED"
        print(f"🔄 FTT Test-time Adaptation: {mode_str}")
        if enable:
            print("   ├── Fast statistics: Will adapt with conservative momentum (0.95)")
            print("   ├── Slow statistics: Will adapt with very conservative momentum (0.995)")
            print("   ├── Ultra statistics: Minimal adaptation (0.9995, no correlation updates)")
            print("   └── Feature importance: Frozen (no updates during inference)")
        else:
            print("   └── All statistics: Frozen during inference")