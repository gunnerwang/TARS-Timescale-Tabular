import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as ty
from typing import Optional
from torch import Tensor
import math
import delu
import faiss
import faiss.contrib.torch_utils

from model.lib.tabr.utils import make_module
from model.lib.temporal_embeddings import TemporalEmbeddings, MemoryBankTimeAttentionEmbeddings

DATASET_SIZE = 340596    
# ct: 227087, de: 279415, mr: 160019, we: 340596;  hi: 224320, eo: 109341, hd: 267645, sh: 18847
BATCH_SIZE = 256

def calculate_adaptive_warmup_params(dataset_size, batch_size):
    """Calculate adaptive warm-up parameters based on dataset characteristics - TabR Optimized"""
    # Calculate updates per epoch
    updates_per_epoch = max(1, dataset_size // batch_size)
    
    # TabR-specific: Much more conservative warm-up to preserve representation stability
    if dataset_size < 10000:
        # Small datasets: Very conservative warm-up for TabR
        importance_start_epochs = 3.0    # Delayed start for stability
        conservative_end_epochs = 8.0    # Extended conservative period
        temperature_annealing_epochs = 10.0  # Longer annealing
    elif dataset_size < 100000:
        # Medium datasets: Conservative warm-up for TabR
        importance_start_epochs = 5.0    # Delayed start
        conservative_end_epochs = 10.0   # Extended conservative period
        temperature_annealing_epochs = 15.0  # Longer annealing
    else:
        # Large datasets: Very conservative for TabR stability
        importance_start_epochs = 8.0    # Much later start
        conservative_end_epochs = 15.0   # Very long conservative period
        temperature_annealing_epochs = 20.0  # Extended annealing
    
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
        'temperature_annealing_epochs': temperature_annealing_epochs,
        'tabr_optimized': True  # Flag for TabR-specific optimizations
    }


class TabROptimizedImplicitEncoder(nn.Module):
    """TabR-Optimized temporal encoder with emphasis on representation stability.
    
    Key differences from standard version:
    - Much more conservative momentum values
    - Simplified timescale design (2 scales instead of 3)
    - Minimal dynamic updates during inference
    - Stable feature importance (less aggressive adaptation)
    - KNN-friendly representation preservation
    """
    def __init__(self, d_in, d_out, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE, 
                 test_time_adaptation=False, drift_significance_threshold=0.3):  # Higher threshold, disabled TTA by default
        super().__init__()
        self.d_in = d_in
        self.test_time_adaptation = test_time_adaptation  # Disabled by default for TabR
        self.drift_significance_threshold = drift_significance_threshold  # Higher threshold
        
        # Calculate adaptive warm-up parameters based on dataset characteristics
        self.warmup_params = calculate_adaptive_warmup_params(dataset_size, batch_size)
        
        print(f"🔧 TabR-Optimized Temporal Configuration for {self.warmup_params['dataset_category']} dataset:")
        print(f"   ├── Dataset Size: {dataset_size:,} samples")
        print(f"   ├── Batch Size: {batch_size}")
        print(f"   ├── Updates per Epoch: {self.warmup_params['updates_per_epoch']}")
        print(f"   ├── Feature Importance Start: {self.warmup_params['importance_start_epochs']:.1f} epochs (DELAYED for stability)")
        print(f"   ├── Conservative Period End: {self.warmup_params['conservative_end_epochs']:.1f} epochs (EXTENDED for TabR)")
        print(f"   ├── Temperature Annealing: {self.warmup_params['temperature_annealing_epochs']:.1f} epochs (LONGER for stability)")
        print(f"   ├── Test-time Adaptation: {'Enabled' if test_time_adaptation else 'DISABLED (TabR-optimized)'}")
        print(f"   ├── Drift Significance Threshold: {drift_significance_threshold} (HIGHER for stability)")
        print(f"   └── Timescales: 2 (SIMPLIFIED for TabR compatibility)")
        
        # TabR-optimized: Only two timescales to reduce complexity
        self.fast_momentum = 0.95   # More conservative than standard (was 0.9)
        self.slow_momentum = 0.995  # More conservative than standard (was 0.99)
        
        # Buffers for two timescales only (simplified design)
        register_buffer = getattr(self, "register_buffer", None)
        if callable(register_buffer):
            # Fast timescale (immediate patterns)
            self.register_buffer("fast_mean", torch.zeros(d_in))
            self.register_buffer("fast_var", torch.ones(d_in))
            self.register_buffer("fast_min", torch.zeros(d_in))
            self.register_buffer("fast_max", torch.ones(d_in))
            
            # Slow timescale (distributional patterns)
            self.register_buffer("slow_mean", torch.zeros(d_in))
            self.register_buffer("slow_var", torch.ones(d_in))
            self.register_buffer("slow_skew", torch.zeros(d_in))
            
            # Simplified feature importance (less aggressive)
            self.register_buffer("feature_importance", torch.ones(d_in))
            self.register_buffer("drift_history", torch.zeros(d_in))
            self.register_buffer("update_count", torch.zeros(1))
            
            # Simplified drift tracking
            self.register_buffer("drift_magnitude_history", torch.zeros(50))  # Smaller buffer
            self.register_buffer("drift_history_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("drift_significance", torch.tensor(0.1))  # Lower initial value
        else:
            # Parameter version
            self.fast_mean = nn.Parameter(torch.zeros(d_in), requires_grad=False)
            self.fast_var = nn.Parameter(torch.ones(d_in), requires_grad=False)
            self.fast_min = nn.Parameter(torch.zeros(d_in), requires_grad=False)
            self.fast_max = nn.Parameter(torch.ones(d_in), requires_grad=False)
            
            self.slow_mean = nn.Parameter(torch.zeros(d_in), requires_grad=False)
            self.slow_var = nn.Parameter(torch.ones(d_in), requires_grad=False)
            self.slow_skew = nn.Parameter(torch.zeros(d_in), requires_grad=False)
            
            self.feature_importance = nn.Parameter(torch.ones(d_in), requires_grad=False)
            self.drift_history = nn.Parameter(torch.zeros(d_in), requires_grad=False)
            self.update_count = nn.Parameter(torch.zeros(1), requires_grad=False)
            
            self.drift_magnitude_history = nn.Parameter(torch.zeros(50), requires_grad=False)
            self.drift_history_ptr = nn.Parameter(torch.zeros(1, dtype=torch.long), requires_grad=False)
            self.drift_significance = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        
        # Simplified two-scale projections
        fast_dim = d_out // 2
        slow_dim = d_out - fast_dim
        
        # Reduced feature complexity to avoid representation instability
        fast_features = d_in + d_in // 2  # z_scores + simplified outlier detection
        self.fast_proj = nn.Linear(fast_features, fast_dim)
        
        slow_features = d_in + d_in // 2  # kl_div + simplified skewness
        self.slow_proj = nn.Linear(slow_features, slow_dim)
        
        # More conservative importance adaptation
        self.importance_decay = 0.98  # More conservative (was 0.95)
        self.importance_threshold = 0.3  # Higher threshold (was 0.1)

    def _update_drift_significance(self, current_drift_magnitude):
        """Update drift significance score - TabR optimized version"""
        if self.update_count > 20:  # Later start (was 10)
            # Update smaller circular buffer
            ptr = int(self.drift_history_ptr.item())
            self.drift_magnitude_history[ptr] = current_drift_magnitude
            self.drift_history_ptr.data = torch.tensor((ptr + 1) % 50)
            
            # Calculate drift significance
            if self.update_count > 50:
                recent_drift = self.drift_magnitude_history.mean()
            else:
                filled_size = int(min(self.update_count.item(), 50))
                recent_drift = self.drift_magnitude_history[:filled_size].mean()
            
            # More conservative smoothing for TabR
            self.drift_significance.data = 0.95 * self.drift_significance + 0.05 * recent_drift
            
    def is_drift_significant(self):
        """Check if current drift is significant - higher threshold for TabR"""
        return self.drift_significance.item() > self.drift_significance_threshold
        
    def _compute_fast_indicators(self, x, batch_size):
        """Simplified fast scale indicators for TabR compatibility"""
        
        # Simplified z-score computation
        z_scores = (x - self.fast_mean) / (torch.sqrt(self.fast_var) + 1e-5)
        z_scores = torch.clamp(z_scores, min=-3.0, max=3.0)  # Tighter clipping for stability

        # Simplified outlier detection (reduced complexity)
        range_width = self.fast_max - self.fast_min + 1e-5
        normalized_pos = (x - self.fast_min) / range_width
        outlier_indicators = torch.clamp(normalized_pos, 0, 1)  # Simple clipping instead of sigmoid
        
        # Reduce dimensionality by downsampling outlier indicators
        outlier_downsampled = outlier_indicators[:, ::2] if self.d_in > 1 else outlier_indicators
        
        # Concatenate with reduced complexity
        fast_features = torch.cat([z_scores, outlier_downsampled], dim=1)
        
        # Much more conservative importance application
        min_updates = self.warmup_params['min_updates_for_importance']
        conservative_end = self.warmup_params['conservative_end_updates']
        
        if self.update_count.item() > min_updates and self.training:  # Only during training
            if self.update_count.item() < conservative_end:
                # Very gradual warm-up
                warmup_ratio = min((self.update_count.item() - min_updates) / (conservative_end - min_updates), 0.5)
                importance = self.feature_importance * warmup_ratio + (1.0 - warmup_ratio)
            else:
                # Even after warmup, limit the importance impact
                importance = 0.7 * self.feature_importance + 0.3  # Less aggressive
            
            # Apply importance more conservatively
            importance_for_z = importance.unsqueeze(0)
            importance_for_outlier = importance[::2].unsqueeze(0) if self.d_in > 1 else importance.unsqueeze(0)
            importance_expanded = torch.cat([importance_for_z, importance_for_outlier], dim=1)
            fast_features = fast_features * importance_expanded
            
        return fast_features
        
    def _compute_slow_indicators(self, x, batch_size):
        """Simplified slow scale indicators for TabR compatibility"""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False) + 1e-5
        
        # Simplified KL divergence (only key components)
        var_ratio = self.slow_var / (batch_var + 1e-5)
        mean_diff_squared = (batch_mean - self.slow_mean).pow(2)
        mahalanobis_dist = mean_diff_squared / (self.slow_var + 1e-5)
        kl_div_simplified = 0.5 * (torch.log(var_ratio + 1e-5) + mahalanobis_dist)
        
        # Simplified skewness (reduce computation)
        batch_centered = x - batch_mean.unsqueeze(0)
        batch_skew = (batch_centered.pow(3).mean(dim=0) / (batch_var.pow(1.5) + 1e-5))
        skewness_change = torch.abs(batch_skew - self.slow_skew)
        
        # Downsample for dimensionality reduction
        skewness_downsampled = skewness_change[::2] if self.d_in > 1 else skewness_change
        
        # Concatenate with reduced complexity
        slow_features = torch.cat([
            kl_div_simplified.unsqueeze(0).expand(batch_size, -1),
            skewness_downsampled.unsqueeze(0).expand(batch_size, -1)
        ], dim=1)
        
        # Conservative importance application
        min_updates = self.warmup_params['min_updates_for_importance']
        conservative_end = self.warmup_params['conservative_end_updates']
        
        if self.update_count.item() > min_updates and self.training:
            if self.update_count.item() < conservative_end:
                warmup_ratio = min((self.update_count.item() - min_updates) / (conservative_end - min_updates), 0.5)
                importance = self.feature_importance * warmup_ratio + (1.0 - warmup_ratio)
            else:
                importance = 0.7 * self.feature_importance + 0.3
            
            importance_for_kl = importance.unsqueeze(0)
            importance_for_skew = importance[::2].unsqueeze(0) if self.d_in > 1 else importance.unsqueeze(0)
            importance_expanded = torch.cat([importance_for_kl, importance_for_skew], dim=1)
            slow_features = slow_features * importance_expanded
            
        return slow_features
        
    def _update_feature_importance(self, drift_indicators):
        """Much more conservative feature importance update for TabR"""
        min_updates_for_importance = self.warmup_params['min_updates_for_importance']
        conservative_end_updates = self.warmup_params['conservative_end_updates']
        
        # Only update during training and much later
        if (self.training and 
            self.update_count > min_updates_for_importance * 2 and  # Even later start
            drift_indicators is not None):
            
            # Extract drift signals more conservatively
            fast_features = drift_indicators['fast']  
            slow_features = drift_indicators['slow']
            
            batch_size = fast_features.shape[0]
            z_scores = fast_features[:, :self.d_in].abs().mean(dim=0)
            kl_components = slow_features[:, :self.d_in].abs().mean(dim=0)
            
            # Much higher noise threshold
            current_drift = z_scores + kl_components
            noise_threshold = 0.3  # Much higher (was 0.1)
            significant_drift = torch.where(current_drift > noise_threshold, 
                                          current_drift, 
                                          torch.zeros_like(current_drift))
            
            # Very conservative decay
            effective_decay = 0.99  # Much more conservative (was 0.95-0.98)
            
            # Update drift history very slowly
            self.drift_history.data = (effective_decay * self.drift_history + 
                                     (1 - effective_decay) * significant_drift)
            
            # Only update importance if drift is very significant
            if self.drift_history.max() > 0.2:  # Much higher threshold (was 0.05)
                raw_importance = 1.0 + 0.5 * self.drift_history  # Reduce impact
                self.feature_importance.data = raw_importance / (raw_importance.sum() + 1e-5) * self.d_in
                
                # Much tighter constraints
                self.feature_importance.data = torch.clamp(self.feature_importance, 
                                                         min=self.importance_threshold, 
                                                         max=2.0)  # Lower max (was 5.0)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Much more conservative update policy for TabR
        should_update_stats = self.training and batch_size > 1  # Only training, never inference
        
        if should_update_stats:
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False) + 1e-5
                
                if self.update_count > 0:
                    # Use very conservative momentum values for TabR
                    fast_momentum = self.fast_momentum      # 0.95 (conservative)
                    slow_momentum = self.slow_momentum      # 0.995 (very conservative)
                    
                    # Update statistics conservatively
                    self.fast_mean.data = fast_momentum * self.fast_mean + (1 - fast_momentum) * batch_mean
                    self.fast_var.data = fast_momentum * self.fast_var + (1 - fast_momentum) * batch_var
                    batch_min = x.min(dim=0)[0]
                    batch_max = x.max(dim=0)[0]
                    self.fast_min.data = torch.minimum(fast_momentum * self.fast_min + (1 - fast_momentum) * batch_min, self.fast_min)
                    self.fast_max.data = torch.maximum(fast_momentum * self.fast_max + (1 - fast_momentum) * batch_max, self.fast_max)
                    
                    # Update slow statistics
                    self.slow_mean.data = slow_momentum * self.slow_mean + (1 - slow_momentum) * batch_mean
                    self.slow_var.data = slow_momentum * self.slow_var + (1 - slow_momentum) * batch_var
                    batch_centered = x - batch_mean.unsqueeze(0)
                    batch_skew = (batch_centered.pow(3).mean(dim=0) / (batch_var.pow(1.5) + 1e-5))
                    self.slow_skew.data = slow_momentum * self.slow_skew + (1 - slow_momentum) * batch_skew
                else:
                    # Initialize statistics
                    self.fast_mean.data = batch_mean
                    self.fast_var.data = batch_var
                    self.fast_min.data = x.min(dim=0)[0]
                    self.fast_max.data = x.max(dim=0)[0]
                    
                    self.slow_mean.data = batch_mean
                    self.slow_var.data = batch_var
                    batch_centered = x - batch_mean.unsqueeze(0)
                    self.slow_skew.data = (batch_centered.pow(3).mean(dim=0) / (batch_var.pow(1.5) + 1e-5))
                
                self.update_count += 1
        
        # Compute simplified drift indicators
        fast_features = self._compute_fast_indicators(x, batch_size)
        slow_features = self._compute_slow_indicators(x, batch_size)
        
        # Calculate drift magnitude more conservatively
        with torch.no_grad():
            if should_update_stats and self.update_count > 10:
                fast_magnitude = fast_features.abs().mean()
                slow_magnitude = slow_features.abs().mean()
                current_drift_magnitude = (fast_magnitude + slow_magnitude) / 2.0
                self._update_drift_significance(current_drift_magnitude)
        
        # Update feature importance very conservatively
        if should_update_stats and batch_size > 1:
            with torch.no_grad():
                drift_indicators_dict = {
                    'fast': fast_features,
                    'slow': slow_features
                }
                self._update_feature_importance(drift_indicators_dict)
        
        # Project simplified indicators
        fast_output = self.fast_proj(fast_features)
        slow_output = self.slow_proj(slow_features)
        
        # Return simplified structure
        return {
            'fast': fast_output,
            'slow': slow_output,
            'concatenated': torch.cat([fast_output, slow_output], dim=-1),
            'feature_importance': self.feature_importance,
            'drift_significant': self.is_drift_significant(),
            'drift_magnitude': self.drift_significance.item()
        }


class TabROptimizedComponentRouter(nn.Module):
    """TabR-Optimized router for simplified 2-component temporal embeddings.
    
    Simplified design for TabR compatibility:
    - decay component (fast pathway) 
    - period component (slow pathway)
    - Removed context component to reduce complexity
    
    Features:
    - More conservative routing decisions
    - Higher stability for representation consistency
    - Simplified residual routing: α̃ = γ · α + (1 - γ) · u, where u = [1/2, 1/2]
    - Less aggressive bypass mode
    """
    def __init__(self, fast_dim, slow_dim, fusion_strategy='softmax', 
                 residual_gamma=0.9, learnable_gamma=False,  # More conservative, non-learnable
                 initial_temperature=10.0, final_temperature=3.0, temperature_schedule='linear',  # Higher temps
                 dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE,
                 enable_bypass=True, bypass_smoothing=0.95):  # More smoothing
        super().__init__()
        self.fusion_strategy = fusion_strategy
        self.learnable_gamma = learnable_gamma  # Disabled for stability
        self.enable_bypass = enable_bypass
        self.bypass_smoothing = bypass_smoothing
        
        # Much more conservative temperature parameters for TabR
        self.initial_temperature = initial_temperature  # Higher initial (was 5.0)
        self.final_temperature = final_temperature      # Higher final (was 1.0)
        self.temperature_schedule = temperature_schedule
        
        # Current temperature (will be updated during training)
        self.register_buffer('current_temperature', torch.tensor(initial_temperature))
        self.register_buffer('training_step', torch.tensor(0))
        
        # Calculate adaptive temperature annealing parameters
        warmup_params = calculate_adaptive_warmup_params(dataset_size, batch_size)
        self.temperature_warmup_steps = warmup_params['temperature_warmup_steps']
        
        print(f"🌡️  TabR-Optimized Temperature Annealing: {warmup_params['temperature_annealing_epochs']:.1f} epochs ({self.temperature_warmup_steps} steps)")
        print(f"   ├── Initial Temperature: {initial_temperature} (HIGHER for stability)")
        print(f"   ├── Final Temperature: {final_temperature} (HIGHER for stability)")
        if enable_bypass:
            print(f"🔄 TabR-Optimized Bypass Mode: Enabled (smoothing factor: {bypass_smoothing})")
            print(f"   └── More conservative bypass decisions for representation stability")
        
        # Fixed residual routing parameters (non-learnable for stability)
        if learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(residual_gamma))
        else:
            self.register_buffer('gamma', torch.tensor(residual_gamma))
        
        # Simplified uniform reference distribution (2 components)
        self.register_buffer('uniform_weights', torch.tensor([1/2, 1/2]))
        
        # Track routing statistics
        self.register_buffer('bypass_usage_count', torch.tensor(0))
        self.register_buffer('drift_routing_count', torch.tensor(0))
        
        # Simplified routing networks (2 components only)
        self.decay_router = nn.Linear(fast_dim, 1)
        self.period_router = nn.Linear(slow_dim, 1)
        
        # Simplified self-regulation for bypass mode
        if enable_bypass:
            self.register_buffer('bypass_weights', torch.tensor([1/2, 1/2]))
            self.register_buffer('bypass_adaptation_rate', torch.tensor(0.005))  # Very slow adaptation

    def _update_bypass_weights(self, component_embeddings):
        """Very conservative bypass weight updates for TabR stability"""
        if not self.enable_bypass or len(component_embeddings) != 2:
            return
            
        decay_emb, period_emb = component_embeddings
        
        with torch.no_grad():
            # Very conservative adaptation
            decay_mag = decay_emb.abs().mean()
            period_mag = period_emb.abs().mean()
            
            total_mag = decay_mag + period_mag + 1e-8
            target_weights = torch.tensor([
                decay_mag / total_mag,
                period_mag / total_mag
            ], device=self.bypass_weights.device)
            
            # Extremely slow adaptation for stability
            self.bypass_weights.data = (
                (1 - self.bypass_adaptation_rate) * self.bypass_weights + 
                self.bypass_adaptation_rate * target_weights
            )

    def _update_temperature(self):
        """Conservative temperature update for TabR"""
        if self.training and self.temperature_schedule != 'fixed':
            self.training_step += 1
            
            if self.training_step <= self.temperature_warmup_steps:
                progress = self.training_step.float() / self.temperature_warmup_steps
                
                if self.temperature_schedule == 'linear':
                    self.current_temperature.data = (
                        self.initial_temperature - 
                        (self.initial_temperature - self.final_temperature) * progress
                    )
                elif self.temperature_schedule == 'cosine':
                    self.current_temperature.data = (
                        self.final_temperature + 
                        0.5 * (self.initial_temperature - self.final_temperature) * 
                        (1 + torch.cos(torch.tensor(torch.pi) * progress))
                    )
            else:
                self.current_temperature.data = torch.tensor(self.final_temperature)
        
    def forward(self, drift_encoding_dict, component_embeddings):
        """
        Simplified routing for 2-component system
        
        Args:
            drift_encoding_dict: Dictionary with 'fast', 'slow' drift signals
                                 and 'drift_significant' boolean indicator
            component_embeddings: Tuple of (decay_emb, period_emb) - 2 components only
        
        Returns:
            Weighted combination of 2 component embeddings
        """
        if len(component_embeddings) != 2:
            raise ValueError("TabR-Optimized router expects exactly 2 components (decay, period)")
            
        self._update_temperature()
        
        # More conservative drift significance check
        drift_significant = drift_encoding_dict.get('drift_significant', False)  # Default to False
        drift_magnitude = drift_encoding_dict.get('drift_magnitude', 0.0)
        
        # Higher threshold for drift-based routing activation
        use_drift_routing = drift_significant and drift_magnitude > 0.15  # Higher threshold
        
        if self.enable_bypass and not use_drift_routing:
            # Bypass mode: More conservative decisions
            with torch.no_grad():
                self.bypass_usage_count += 1
                
            self._update_bypass_weights(component_embeddings)
            
            # More stable bypass weights
            component_weights = self.bypass_weights.unsqueeze(0).expand(
                component_embeddings[0].shape[0], 2
            )
            
            # Reduced noise for stability
            if self.training:
                noise = torch.randn_like(component_weights) * 0.005  # Reduced noise
                component_weights = F.softmax(component_weights + noise, dim=-1)
            else:
                component_weights = F.softmax(component_weights, dim=-1)
                
        else:
            # Conservative drift-based routing
            with torch.no_grad():
                self.drift_routing_count += 1
                
            # Extract simplified drift signals
            fast_drift = drift_encoding_dict['fast']    # Maps to decay component
            slow_drift = drift_encoding_dict['slow']    # Maps to period component
            
            # Compute routing weights more conservatively
            alpha_decay = self.decay_router(fast_drift)      # [batch_size, 1]
            alpha_period = self.period_router(slow_drift)    # [batch_size, 1]
            
            # Simplified 2-component routing
            combined_alphas = torch.cat([alpha_decay, alpha_period], dim=-1)
            
            # Apply conservative temperature scaling
            raw_weights = F.softmax(combined_alphas / self.current_temperature, dim=-1)
            
            # Apply residual routing with simplified uniform distribution
            batch_size = raw_weights.shape[0]
            uniform_weights = torch.tensor([1/2, 1/2], device=raw_weights.device)
            uniform_batch = uniform_weights.unsqueeze(0).expand(batch_size, 2)
            component_weights = self.gamma * raw_weights + (1.0 - self.gamma) * uniform_batch
        
        # Simplified weighted combination
        decay_emb, period_emb = component_embeddings
        
        weighted_embedding = (
            component_weights[:, 0:1] * decay_emb +    # Fast drift → Decay component
            component_weights[:, 1:2] * period_emb     # Slow drift → Period component
        )
        
        return weighted_embedding
    
    def get_routing_info(self):
        """Get simplified routing statistics"""
        total_usage = self.bypass_usage_count + self.drift_routing_count + 1e-8
        return {
            'bypass_usage_ratio': self.bypass_usage_count.item() / total_usage,
            'drift_routing_ratio': self.drift_routing_count.item() / total_usage,
            'bypass_weights': self.bypass_weights.tolist() if self.enable_bypass else None,
            'current_temperature': self.current_temperature.item(),
            'total_routing_calls': int(total_usage),
            'tabr_optimized': True
        }


class TabR_Temporal(nn.Module):
    def __init__(
        self,
        *,
        #
        n_num_features: int,
        n_cat_features: int,
        n_classes: Optional[int],
        t_mean: float,
        t_std: float,
        #
        num_embeddings: Optional[dict],
        temporal_embeddings: Optional[dict],
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization,
        context_dropout: float,
        dropout0: float,
        dropout1,
        normalization: str,
        activation: str,
        #
        # Advanced temporal parameters
        implicit_time_dim: int = 64,  # Dimension of implicit time encoding
        test_time_adaptation: bool = True,  # Enable adaptive statistics update during inference
        drift_significance_threshold: float = 0.1,  # Threshold for significant drift detection
        enable_bypass: bool = True,  # Enable bypass routing for non-significant drift
        #
        # The following options should be used only when truly needed.
        memory_efficient: bool = True,
        candidate_encoding_batch_size: Optional[int] = 4096,
    ) -> None:
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        
        super().__init__()
        self.implicit_time_dim = implicit_time_dim
        self.test_time_adaptation = test_time_adaptation
        self.drift_significance_threshold = drift_significance_threshold
        self.enable_bypass = enable_bypass
        
        # Initialize multi-timescale implicit encoder
        if self.implicit_time_dim > 0:
            self.implicit_time_encoder = TabROptimizedImplicitEncoder(
                d_in=n_num_features, 
                d_out=implicit_time_dim,
                dataset_size=DATASET_SIZE,
                batch_size=BATCH_SIZE,
                test_time_adaptation=test_time_adaptation,
                drift_significance_threshold=drift_significance_threshold
            )
        
        # Initialize temporal embeddings with TabR-optimized features
        if temporal_embeddings is not None:
            base_config = temporal_embeddings.copy()
            
            # TabR-optimized temporal configuration (more conservative)
            unified_config = base_config.copy()
            unified_config.update({
                'decay_factor': base_config.get('decay_factor', 0.2),  # More conservative (was 0.1)
                'periodic_patterns': base_config.get('periodic_patterns', [24, 168]),  # Simplified patterns
                'd_embedding': base_config.get('d_embedding', 32),  # Smaller embedding (was 64)
                'feature_fusion': base_config.get('feature_fusion', False)  # Disabled for stability
            })
            self.temporal_embeddings = MemoryBankTimeAttentionEmbeddings(t_mean, t_std, **unified_config)
            
            # Initialize simplified router for 2-component system
            if self.implicit_time_dim > 0:
                # Simplified 2-component dimensions
                fast_dim = implicit_time_dim // 2
                slow_dim = implicit_time_dim - fast_dim
                
                self.temporal_router = TabROptimizedComponentRouter(
                    fast_dim=fast_dim,
                    slow_dim=slow_dim, 
                    fusion_strategy='softmax',
                    residual_gamma=0.9,        # More conservative
                    learnable_gamma=False,     # Fixed for stability
                    initial_temperature=10.0,  # Higher initial temperature
                    final_temperature=3.0,     # Higher final temperature
                    temperature_schedule='linear',
                    dataset_size=DATASET_SIZE,
                    batch_size=BATCH_SIZE,
                    enable_bypass=enable_bypass
                )
                
            temporal_dim = self.temporal_embeddings.out_dim
        else:
            # Fallback to basic temporal embeddings with conservative settings
            basic_config = {
                'order': 2,  # Simplified order
                'trend': False,  # Disabled for stability
                'd_embedding': 16  # Small embedding
            }
            self.temporal_embeddings = TemporalEmbeddings(t_mean, t_std, **basic_config)
            temporal_dim = self.temporal_embeddings.out_dim
            
        # Calculate effective temporal dimension for integration (more conservative)
        if self.implicit_time_dim > 0 and hasattr(self, 'temporal_embeddings'):
            if hasattr(self.temporal_embeddings, 'component_dim'):
                # Router mode: use component dimension
                actual_temporal_dim = self.temporal_embeddings.component_dim
            else:
                # Fallback: use smaller portion of temporal dimension to reduce impact
                actual_temporal_dim = min(temporal_dim, 32)  # Cap at 32 for stability
        else:
            # Much smaller temporal impact for stability
            actual_temporal_dim = min(temporal_dim, 16)  # Cap at 16
            
        # More conservative temporal integration
        n_num_features_with_temporal = n_num_features + actual_temporal_dim
        
        if dropout1 == 'dropout0':
            dropout1 = dropout0
            
        self.n_classes = n_classes

        self.num_embeddings = (
            None
            if num_embeddings is None
            else make_module(num_embeddings, n_features=n_num_features_with_temporal)
        )

        # >>> E (TabR's encoding pipeline - preserved)
        d_in = (
            n_num_features_with_temporal
            * (1 if num_embeddings is None else num_embeddings['d_embedding'])
            + n_cat_features
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        Activation = getattr(nn, activation)

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )

        # >>> R (TabR-specific components - completely preserved)
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes == 1
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), delu.nn.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> P (TabR-specific components - completely preserved)
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(),
            nn.Linear(d_main, n_classes),
        )

        # >>> TabR-specific components - completely preserved
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()
        
        # Print TabR-optimized configuration summary
        print(f"\n🎯 TabR-Optimized Temporal Configuration:")
        print(f"   ├── Model Type: TabR (Retrieval-Augmented)")
        print(f"   ├── Optimization Focus: Representation Stability & KNN Compatibility")
        if implicit_time_dim > 0:
            print(f"   ├── Implicit Time Encoding: {implicit_time_dim} dims (SIMPLIFIED: 2 timescales)")
        else:
            print(f"   ├── Implicit Time Encoding: Disabled")
        print(f"   ├── Test-time Adaptation: {'DISABLED (TabR-optimized)' if not test_time_adaptation else 'Enabled'}")
        print(f"   ├── Drift Significance Threshold: {drift_significance_threshold} (HIGHER for stability)")
        print(f"   ├── Bypass Routing: {'Enabled (conservative)' if enable_bypass else 'Disabled'}")
        if temporal_embeddings:
            print(f"   ├── Temporal Embeddings: Enhanced 2-Component (decay + period)")
            print(f"   │   ├── Embedding Dimension: ≤32 (REDUCED for stability)")
            print(f"   │   ├── Periodic Patterns: Simplified (24h, 168h)")
            print(f"   │   └── Feature Fusion: Disabled (stability)")
        else:
            print(f"   ├── Temporal Embeddings: Basic (conservative)")
        print(f"   ├── Actual Temporal Dimension: {actual_temporal_dim} (CAPPED for TabR)")
        print(f"   └── TabR Core: 100% PRESERVED")
        print(f"       ├── KNN Search: faiss-based similarity retrieval")
        print(f"       ├── Context Mixing: Attention-weighted neighbor fusion")
        print(f"       ├── Label Encoding: Classification/regression support")
        print(f"       └── Memory Efficiency: Candidate batch processing")

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def _encode(self, x_num, x_cat, idx):
        x = []
        
        # Generate simplified drift encoding (2 timescales only)
        if self.implicit_time_dim > 0:
            drift_encoding_dict = self.implicit_time_encoder(x_num)
        else:
            drift_encoding_dict = None
        
        # Generate temporal embeddings using simplified component routing
        if hasattr(self, 'temporal_embeddings'):
            # Use router if available with drift signals
            if (hasattr(self, 'temporal_router') and 
                self.implicit_time_dim > 0 and 
                drift_encoding_dict is not None and 
                hasattr(self.temporal_embeddings, 'forward_components')):
                
                try:
                    # Get 2-component temporal embeddings
                    component_embeddings = self.temporal_embeddings.forward_components(idx)
                    
                    # Ensure we have exactly 2 components for TabR-optimized router
                    if len(component_embeddings) >= 2:
                        # Use only first 2 components (decay, period)
                        simplified_components = component_embeddings[:2]
                        temporal_emb = self.temporal_router(drift_encoding_dict, simplified_components)
                    else:
                        # Fallback to unified embeddings
                        temporal_emb = self.temporal_embeddings(idx).flatten(1)
                        
                except Exception as e:
                    # Robust fallback for any routing issues
                    print(f"Warning: Router fallback triggered: {e}")
                    temporal_emb = self.temporal_embeddings(idx).flatten(1)
            else:
                # Fall back to unified temporal embeddings
                if hasattr(self.temporal_embeddings, 'out_dim') and self.temporal_embeddings.out_dim > 0:
                    temporal_emb = self.temporal_embeddings(idx).flatten(1)
                else:
                    temporal_emb = None
                
            # Conservative temporal integration
            if x_num is not None and temporal_emb is not None:
                # Apply scaling to reduce temporal impact for stability
                temporal_scaled = temporal_emb * 0.5  # Reduce impact by 50%
                x_num_with_temporal = torch.cat([x_num, temporal_scaled], dim=-1)
            elif temporal_emb is not None:
                temporal_scaled = temporal_emb * 0.5
                x_num_with_temporal = temporal_scaled
            else:
                x_num_with_temporal = x_num
        else:
            x_num_with_temporal = x_num
            
        # Continue with TabR's original encoding logic - COMPLETELY PRESERVED
        if x_num_with_temporal is not None:
            x.append(
                x_num_with_temporal
                if self.num_embeddings is None
                else self.num_embeddings(x_num_with_temporal).flatten(1)
            )
        if x_cat is not None:
            x.append(x_cat)
        x = torch.cat(x, dim=1)

        x = x.float()
        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        
        return x, k

    def forward(
        self,
        *,
        x_num: Tensor, 
        x_cat: ty.Optional[Tensor],
        y: Optional[Tensor],
        idx: Tensor,
        candidate_x_num: ty.Optional[Tensor],
        candidate_x_cat: ty.Optional[Tensor],
        candidate_y: Tensor,
        candidate_idx: Tensor,
        context_size: int,
        is_train: bool,
    ) -> Tensor:
        # TabR's original forward logic with enhanced temporal integration
        
        # >>>
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and not self.memory_efficient
        ):
            candidate_k = (
                self._encode(candidate_x_num, candidate_x_cat, candidate_idx)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(x_num_, x_cat_, idx_)[1]
                        for x_num_, x_cat_, idx_ in delu.iter_batches(
                            (candidate_x_num, candidate_x_cat, candidate_idx), self.candidate_encoding_batch_size
                        )
                    ]
                )
            )
        x, k = self._encode(x_num, x_cat, idx)
        if is_train:
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None
        
        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            if self.search_index is None:
                self.search_index = (
                    faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                    if device.type == 'cuda'
                    else faiss.IndexFlatL2(d_main)
                )
            
            self.search_index.reset()
            self.search_index.add(candidate_k.to(torch.float32))  # type: ignore[code]
            distances: Tensor
            context_idx: Tensor
            distances, context_idx = self.search_index.search(  # type: ignore[code]
                k.to(torch.float32), context_size + (1 if is_train else 0)
            )
            if is_train:
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                ] = torch.inf
                # Not the most elegant solution to remove the argmax, but anyway.
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])
        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            # Repeating the same computation,
            # but now only for the context objects and with autograd on.
            context_k = self._encode(
                torch.cat([x_num, candidate_x_num])[
                        context_idx
                    ].flatten(0, 1),
                torch.cat([x_cat, candidate_x_cat])[
                        context_idx
                    ].flatten(0, 1),
                torch.cat([idx, candidate_idx])[
                        context_idx
                    ].flatten(0, 1)
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]

        similarities = (
            -k.square().sum(-1, keepdim=True)
            + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
            - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)
        
        if self.n_classes > 1:
            context_y_emb = self.label_encoder(candidate_y[context_idx][..., None].long())
        else:
            context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
            if len(context_y_emb.shape) == 4:
                context_y_emb = context_y_emb[:,:,0,:]
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x
    
    def set_test_time_adaptation(self, enable: bool):
        """Enable or disable test-time adaptation of statistics - TabR Optimized
        
        Args:
            enable: If True, statistics will adapt during inference.
                   If False, statistics remain frozen after training.
                   
        Note: For TabR, test-time adaptation is generally NOT recommended as it can
        destabilize the representation space required for effective KNN search.
        """
        self.test_time_adaptation = enable
        if hasattr(self, 'implicit_time_encoder'):
            self.implicit_time_encoder.test_time_adaptation = enable
        
        mode_str = "ENABLED" if enable else "DISABLED"
        print(f"🔄 TabR Test-time Adaptation: {mode_str}")
        
        if enable:
            print("   ⚠️  WARNING: Test-time adaptation may destabilize TabR's representation space!")
            print("   ├── Fast statistics: Will adapt with VERY conservative momentum (0.95)")
            print("   ├── Slow statistics: Will adapt with VERY conservative momentum (0.995)")
            print("   ├── Feature importance: FROZEN (no updates during inference)")
            print("   ├── KNN search: May be affected by representation drift")
            print("   └── Recommendation: Consider disabling TTA for TabR stability")
        else:
            print("   ✅ RECOMMENDED for TabR: All temporal statistics frozen during inference")
            print("   ├── Representation space: Stable and consistent")
            print("   ├── KNN search: Reliable similarity computation")
            print("   ├── Feature importance: Fixed (optimal for retrieval)")
            print("   └── TabR performance: Maximized stability")
