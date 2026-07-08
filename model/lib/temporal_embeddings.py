import torch
import torch.nn as nn
from torch import Tensor
import math
from typing import Optional, List
import torch.nn.functional as F


class FourierEmbeddings(nn.Module):
    def __init__(
        self, period: float, order: int
    ) -> None:
        super().__init__()
        self.frequencies = torch.arange(1, order + 1, dtype=torch.float32) / period
        self.frequencies = nn.Parameter(self.frequencies)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 1
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return x


class TemporalEmbeddings(nn.Module):
    def __init__(
        self,
        t_mean: float,
        t_std: float,
        order: List[int],
        trend: bool,
        d_embedding: int,
    ) -> None:
        super().__init__()
        self.order = order
        self.trend = trend
        self.periodicity = sum(order) > 0
        self.out_dim = (d_embedding if self.periodicity else 0) + (1 if self.trend else 0)
        
        if self.trend:
            self.t_mean = t_mean
            self.t_std = t_std
        
        if self.periodicity:
            assert len(order) == 4, "The length of orders must be 4, corresponding to (year, month, week, day)"
            # Create embeddings only for non-zero orders
            embeddings = []
            if order[0] > 0:
                embeddings.append(FourierEmbeddings(31557600.0, order[0]))  # year
            if order[1] > 0:
                embeddings.append(FourierEmbeddings(2629800.0, order[1]))   # month
            if order[2] > 0:
                embeddings.append(FourierEmbeddings(604800.0, order[2]))    # week
            if order[3] > 0:
                embeddings.append(FourierEmbeddings(86400.0, order[3]))     # day
            
            self.embeddings = nn.ModuleList(embeddings)
            self.linear = nn.Linear(2 * sum(order), d_embedding)
            self.relu = nn.ReLU()

    def forward(self, x):
        x_trend = (x[..., None] - self.t_mean) / self.t_std if self.trend else None
        if self.periodicity:
            x = torch.cat([module(x) for module in self.embeddings], dim=-1)
            x = self.linear(x)
            x = self.relu(x)
            x = torch.cat([x, x_trend], dim=-1) if x_trend is not None else x
        else:
            assert x_trend is not None, "forwards() will not be called if self.out_dim == 0"
            x = x_trend
        return x


class RawTemporalEmbeddings(nn.Module):
    def __init__(
        self,
        t_mean: float,
        t_std: float,
        order: List[int],
        trend: bool,
        d_embedding: int,
    ) -> None:
        super().__init__()
        self.order = order
        self.trend = trend
        self.periodicity = sum(order) > 0
        self.out_dim = (d_embedding if self.periodicity else 0) + (1 if self.trend else 0)

        if self.trend:
            self.t_mean = t_mean
            self.t_std = t_std

        if self.periodicity:
            assert len(order) == 4, "The length of orders must be 4, corresponding to (year, month, week, day)"
            self.embeddings = nn.ModuleList([
                FourierEmbeddings(31557600.0, order[0]) if order[0] else None,
                FourierEmbeddings(2629800.0, order[1]) if order[1] else None,
                FourierEmbeddings(604800.0, order[2]) if order[2] else None,
                FourierEmbeddings(86400.0, order[3]) if order[3] else None,
            ])
            self.embeddings = nn.ModuleList([embedding for embedding in self.embeddings if embedding is not None])
            self.linear_in_dim = 2 * sum(order)
            self.linear_out_dim = d_embedding

    def forward(self, x):
        x_period = torch.cat([module(x) for module in self.embeddings], dim=-1) if self.periodicity else None
        x_trend = (x[..., None] - self.t_mean) / self.t_std if self.trend else None
        return x_period, x_trend


class FullTemporalEmbeddings(nn.Module):
    def __init__(
        self,
        t_mean: float,
        t_std: float,
        d_embedding: int,
    ) -> None:
        super().__init__()
        d_embedding = max(d_embedding, 0)
        self.out_dim = d_embedding + 1
        self.t_mean = t_mean
        self.t_std = t_std

        if d_embedding > 0:
            self.embeddings = nn.ModuleList([
                FourierEmbeddings(31557600.0, 128),
                FourierEmbeddings(2629800.0, 128),
                FourierEmbeddings(604800.0, 128),
                FourierEmbeddings(86400.0, 128),
            ])
            self.linear = nn.Linear(2 * 4 * 128, d_embedding)
            self.relu = nn.ReLU()

    def forward(self, x):
        assert self.out_dim > 0, "forwards() will not be called if self.out_dim <= 0"
        x_trend = (x[..., None] - self.t_mean) / self.t_std
        if self.out_dim > 1:
            x = torch.cat([module(x) for module in self.embeddings], dim=-1)
            x = self.linear(x)
            x = self.relu(x)
            x = torch.cat([x, x_trend], dim=-1)
        else:
            x = x_trend
        return x


class FullRawTemporalEmbeddings(nn.Module):
    def __init__(
        self,
        t_mean: float,
        t_std: float,
        d_embedding: int,
    ) -> None:
        super().__init__()
        self.out_dim = d_embedding + 1
        self.t_mean = t_mean
        self.t_std = t_std

        self.embeddings = nn.ModuleList([
            FourierEmbeddings(31557600.0, 128),
            FourierEmbeddings(2629800.0, 128),
            FourierEmbeddings(604800.0, 128),
            FourierEmbeddings(86400.0, 128),
        ])
        self.linear_in_dim = 2 * 4 * 128
        self.linear_out_dim = d_embedding

    def forward(self, x):
        x_period = torch.cat([module(x) for module in self.embeddings], dim=-1)
        x_trend = (x[..., None] - self.t_mean) / self.t_std
        return x_period, x_trend


class TimestampNorm(nn.Module):
    def __init__(
        self,
        t_mean: float,
        t_std: float,
    ) -> None:
        super().__init__()
        
        self.t_mean = t_mean
        self.t_std = t_std

    def forward(self, x):
        x = (x - self.t_mean) / self.t_std
        return x


class PeriodicEmbeddings(nn.Module):
    """ The 'PeriodicEmbeddings' from 'On Embeddings for Numerical Features in Tabular Deep Learning'.
    """
    def __init__(
        self, n_frequencies: int, frequency_scale: float
    ) -> None:
        super().__init__()
        self.frequencies = nn.Parameter(
            torch.normal(0.0, frequency_scale, (1, n_frequencies))
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 1
        x = 2 * torch.pi * self.frequencies * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return x


class TemporalEmbeddings_PLR(nn.Sequential):
    def __init__(
        self,
        t_mean: float,
        t_std: float,
        n_frequencies: int,
        frequency_scale: float,
        d_embedding: int,
    ) -> None:
        super().__init__(
            TimestampNorm(t_mean, t_std),
            PeriodicEmbeddings(n_frequencies, frequency_scale),
            nn.Linear(2 * n_frequencies, d_embedding),
            nn.ReLU(),
        )


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire (LIF) neuron model for spiking neural networks."""
    def __init__(
        self, 
        tau_mem: float = 10.0,      # Membrane time constant
        threshold: float = 1.0,      # Firing threshold
        reset_potential: float = 0.0 # Reset potential after spike
    ) -> None:
        super().__init__()
        self.tau_mem = tau_mem
        self.threshold = threshold
        self.reset_potential = reset_potential
        
    def forward(self, x: Tensor, mem: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        batch_size = x.size(0)
        
        # Initialize membrane potential if not provided
        if mem is None:
            mem = torch.zeros_like(x)
        
        # Update membrane potential with leakage
        decay = torch.exp(torch.tensor(-1.0 / self.tau_mem))
        mem = decay * mem + x
        
        # Check for spikes
        spike = (mem >= self.threshold).float()
        
        # Reset membrane potential for neurons that spiked
        mem = mem * (1 - spike) + self.reset_potential * spike
        
        return spike, mem


class SpikingTemporalEmbeddings(nn.Module):
    """Temporal embeddings using a spiking neural network approach."""
    def __init__(
        self,
        t_mean: float,
        t_std: float,
        n_inputs: int = 10,         # Number of input neurons
        n_hidden: int = 20,         # Number of hidden neurons
        d_embedding: int = 64,      # Output embedding dimension
        simulation_steps: int = 5,  # Number of time steps to simulate
        tau_mem: float = 10.0,      # Membrane time constant
        threshold: float = 1.0,     # Firing threshold
        trend: bool = True          # Whether to include trend information
    ) -> None:
        super().__init__()
        self.t_mean = t_mean
        self.t_std = t_std
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.simulation_steps = simulation_steps
        self.trend = trend
        self.out_dim = d_embedding + (1 if trend else 0)
        
        # Normalize timestamps
        self.norm = TimestampNorm(t_mean, t_std)
        
        # Input encoding - transform scalar timestamp into multiple inputs
        self.input_transform = nn.Linear(1, n_inputs)
        
        # Hidden layer - LIF neurons
        self.hidden_lif = LIFNeuron(tau_mem, threshold)
        self.hidden_weights = nn.Parameter(
            torch.Tensor(n_inputs, n_hidden).normal_(0.0, 0.1)
        )
        
        # Output layer
        self.output_linear = nn.Linear(n_hidden, d_embedding)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        # Get trend component if needed
        x_trend = (x[..., None] - self.t_mean) / self.t_std if self.trend else None
        
        # Normalize timestamp
        x = self.norm(x)
        x = x.unsqueeze(-1)  # Add feature dimension
        
        # Transform scalar input to multiple inputs
        inputs = self.input_transform(x)
        
        # Initialize membrane potentials
        mem_hidden = torch.zeros(x.size(0), self.n_hidden, device=x.device)
        
        # Store spike history
        spike_history = []
        
        # Simulate for multiple time steps
        for _ in range(self.simulation_steps):
            # Forward to hidden layer
            hidden_input = torch.matmul(inputs, self.hidden_weights)
            hidden_spikes, mem_hidden = self.hidden_lif(hidden_input, mem_hidden)
            spike_history.append(hidden_spikes)
        
        # Aggregate spikes over time
        spike_count = torch.stack(spike_history, dim=1).sum(dim=1)
        
        # Convert spike counts to embedding
        embedding = self.output_linear(spike_count)
        embedding = self.activation(embedding)
        
        # Add trend if needed
        if x_trend is not None:
            embedding = torch.cat([embedding, x_trend], dim=-1)
        
        return embedding


class MemoryBank(nn.Module):
    """Dynamic memory bank for temporal context representation with adaptive updates."""
    def __init__(
        self,
        num_slots: int,
        slot_dim: int,
        distance_type: str = "l2",  # "l2" or "l1"
        temperature: float = 1.0,
        # Dynamic update parameters
        update_rate: float = 0.1,
        decay_factor: float = 0.99,
        min_activation_threshold: float = 0.1,
        use_gating: bool = True,
        # Memory replacement parameters
        novelty_threshold: float = 0.7,
        cleanup_time_threshold: float = 100.0,
        cleanup_usage_threshold: float = 0.01,
        # Performance optimization
        update_frequency: int = 10,  # Update memory every N steps
        enable_dynamic_update: bool = True,  # Completely disable updates for inference
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.distance_type = distance_type
        self.temperature = temperature
        self.update_rate = update_rate
        self.decay_factor = decay_factor
        self.min_activation_threshold = min_activation_threshold
        self.use_gating = use_gating
        self.novelty_threshold = novelty_threshold
        self.cleanup_time_threshold = cleanup_time_threshold
        self.cleanup_usage_threshold = cleanup_usage_threshold
        self.update_frequency = update_frequency
        self.enable_dynamic_update = enable_dynamic_update
        
        # Initialize memory slots as buffers (not parameters) for dynamic updates
        self.register_buffer('memory_slots', torch.randn(num_slots, slot_dim) * 0.1)
        self.register_buffer('time_anchors', torch.randn(num_slots) * 0.1)
        
        # Track usage frequency and last access time for each slot
        self.register_buffer('usage_frequency', torch.zeros(num_slots))
        self.register_buffer('last_access_time', torch.zeros(num_slots))
        
        # Track replacement statistics
        self.register_buffer('replacement_count', torch.zeros(num_slots))
        self.register_buffer('total_replacements', torch.tensor(0.0))
        self.register_buffer('update_step_counter', torch.tensor(0))
        
        # Performance tracking
        self.register_buffer('total_forward_calls', torch.tensor(0))
        self.register_buffer('skipped_updates', torch.tensor(0))
        
        # Learnable components for dynamic updates
        self.input_projection = nn.Linear(slot_dim, slot_dim)
        
        if self.use_gating:
            # Gating mechanism to control memory updates
            self.write_gate = nn.Sequential(
                nn.Linear(slot_dim * 2, slot_dim),
                nn.Sigmoid()
            )
            self.erase_gate = nn.Sequential(
                nn.Linear(slot_dim * 2, slot_dim),
                nn.Sigmoid()
            )
        
        # Time encoding for temporal anchors
        self.time_encoder = nn.Linear(1, slot_dim)
        
    def compute_distance(self, t_norm: Tensor, anchors: Tensor) -> Tensor:
        """Compute distance between normalized timestamps and time anchors.
        
        Args:
            t_norm: Normalized timestamps [batch_size]
            anchors: Time anchors [num_slots]
            
        Returns:
            Distance matrix [batch_size, num_slots]
        """
        # Expand dimensions for broadcasting
        t_expanded = t_norm.unsqueeze(1)  # [batch_size, 1]
        anchors_expanded = anchors.unsqueeze(0)  # [1, num_slots]
        
        if self.distance_type == "l2":
            distances = (t_expanded - anchors_expanded) ** 2
        elif self.distance_type == "l1":
            distances = torch.abs(t_expanded - anchors_expanded)
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
            
        return distances
    
    def update_memory(self, input_emb: Tensor, t_norm: Tensor, attention_weights: Tensor):
        """Dynamically update memory slots based on current input.
        
        Args:
            input_emb: Input embedding for matching and storing [batch_size, slot_dim]
            t_norm: Normalized timestamps [batch_size]
            attention_weights: Attention weights [batch_size, num_slots]
        """
        if not self.training or not self.enable_dynamic_update:
            return  # Only update during training and if enabled
            
        # Increment step counter and check if we should update
        with torch.no_grad():
            self.update_step_counter.data += 1
            should_update = (self.update_step_counter % self.update_frequency) == 0
            
        if not should_update:
            with torch.no_grad():
                self.skipped_updates.data += 1
            return  # Skip update to improve performance
            
        batch_size = input_emb.size(0)
        current_time = t_norm.mean().item()  # Use mean time as current time
        
        # Apply temporal decay (less frequent, more efficient)
        with torch.no_grad():
            if self.update_step_counter % (self.update_frequency * 5) == 0:  # Decay every 50 steps
                self.memory_slots.data *= self.decay_factor
            
        # Periodic memory cleanup (even less frequent)
        if self.update_step_counter % (self.update_frequency * 10) == 0:  # Cleanup every 100 steps
            current_time_diff = current_time - self.last_access_time
            very_old_mask = current_time_diff > self.cleanup_time_threshold
            very_low_usage_mask = self.usage_frequency < self.cleanup_usage_threshold
            
            cleanup_candidates = very_old_mask & very_low_usage_mask
            if cleanup_candidates.any():
                representative_input = input_emb.mean(dim=0, keepdim=True)
                candidate_indices = cleanup_candidates.nonzero(as_tuple=True)[0]
                candidate_times = self.last_access_time[candidate_indices]
                oldest_idx = candidate_indices[candidate_times.argmin()].item()
                self.replace_memory_slot(representative_input, current_time, oldest_idx)
        # Simplified batch update for better performance
        max_attention_per_slot = attention_weights.max(dim=0)[0]
        active_mask = max_attention_per_slot > self.min_activation_threshold
        
        # Update frequency and access time for active slots
        with torch.no_grad():
            self.usage_frequency.data += max_attention_per_slot * active_mask.float()
            self.last_access_time.data = torch.where(
                active_mask, 
                torch.full_like(self.last_access_time, current_time), 
                self.last_access_time.data
            )
        
        # Batch update: only update the most activated slots
        max_weights, max_indices = attention_weights.max(dim=1)  # [batch_size]
        valid_updates = max_weights > self.min_activation_threshold
        
        if valid_updates.any():
            # Get valid samples and their target slots
            valid_inputs = input_emb[valid_updates]  # [valid_batch, slot_dim]
            valid_indices = max_indices[valid_updates]  # [valid_batch]
            
            # Vectorized update for better performance
            with torch.no_grad():
                # Group updates by slot to avoid conflicts
                unique_indices, inverse_indices = torch.unique(valid_indices, return_inverse=True)
                
                for slot_idx in unique_indices:
                    slot_idx = slot_idx.item()
                    # Find all inputs targeting this slot
                    mask = valid_indices == slot_idx
                    slot_inputs = valid_inputs[mask]
                    
                    if len(slot_inputs) > 0:
                        # Average multiple inputs targeting the same slot
                        avg_input = slot_inputs.mean(dim=0)
                        current_memory = self.memory_slots[slot_idx]
                        
                        # Update memory slot
                        self.memory_slots.data[slot_idx] = (
                            (1 - self.update_rate) * current_memory + 
                            self.update_rate * avg_input
                        )
                        
                        # Update time anchor
                        self.time_anchors.data[slot_idx] = (
                            (1 - self.update_rate) * self.time_anchors.data[slot_idx] + 
                            self.update_rate * current_time
                        )
    
    def get_least_used_slot(self) -> int:
        """Find the least used memory slot for replacement."""
        # Combine low usage frequency and old access time to find replacement candidate
        time_weight = 0.3
        freq_weight = 0.7
        
        # Normalize metrics
        norm_freq = self.usage_frequency / (self.usage_frequency.max() + 1e-8)
        current_time = self.last_access_time.max()
        time_diff = current_time - self.last_access_time
        norm_time_diff = time_diff / (time_diff.max() + 1e-8)
        
        # Lower score means better candidate for replacement
        replacement_score = freq_weight * (1 - norm_freq) + time_weight * norm_time_diff
        return replacement_score.argmax().item()
    
    def replace_memory_slot(self, new_memory: Tensor, new_time: float, slot_idx: Optional[int] = None):
        """Replace a specific memory slot or the least used slot with new information."""
        if slot_idx is None:
            slot_idx = self.get_least_used_slot()
        
        with torch.no_grad():
            self.memory_slots.data[slot_idx] = new_memory.squeeze()
            self.time_anchors.data[slot_idx] = new_time
            self.usage_frequency.data[slot_idx] = 0.0
            self.last_access_time.data[slot_idx] = new_time
            
            # Update replacement statistics
            self.replacement_count.data[slot_idx] += 1
            self.total_replacements.data += 1
    
    def forward(self, t_norm: Tensor, update_input: Optional[Tensor] = None) -> Tensor:
        """Query memory bank with normalized timestamps and optionally update it.
        
        Args:
            t_norm: Normalized timestamps [batch_size]
            update_input: Optional input for updating memory [batch_size, slot_dim]
            
        Returns:
            Context embeddings [batch_size, slot_dim]
        """
        with torch.no_grad():
            self.total_forward_calls.data += 1
        # Compute distances
        distances = self.compute_distance(t_norm, self.time_anchors)
        
        # Apply softmax attention (negative distance for similarity)
        attention_weights = torch.softmax(-distances / self.temperature, dim=1)
        
        # Weighted combination of memory slots
        context_emb = torch.matmul(attention_weights, self.memory_slots)
        
        # Update memory if input is provided (with frequency control)
        if update_input is not None and self.training and self.enable_dynamic_update:
            # Project input for memory update
            projected_input = self.input_projection(update_input)
            
            # Update memory based on current input (frequency controlled)
            self.update_memory(projected_input, t_norm, attention_weights)
        
        return context_emb
    
    def reset_memory(self):
        """Reset memory bank to initial state."""
        self.memory_slots.data = torch.randn_like(self.memory_slots) * 0.1
        self.time_anchors.data = torch.randn_like(self.time_anchors) * 0.1
        self.usage_frequency.data.fill_(0.0)
        self.last_access_time.data.fill_(0.0)
        self.replacement_count.data.fill_(0.0)
        self.total_replacements.data.fill_(0.0)
        self.update_step_counter.data.fill_(0)
        with torch.no_grad():
            self.total_forward_calls.data.fill_(0)
            self.skipped_updates.data.fill_(0)
    
    def get_memory_stats(self) -> dict:
        """Get statistics about memory usage."""
        return {
            'usage_frequency': self.usage_frequency.cpu().numpy(),
            'last_access_time': self.last_access_time.cpu().numpy(),
            'replacement_count': self.replacement_count.cpu().numpy(),
            'memory_utilization': (self.usage_frequency > 0).float().mean().item(),
            'avg_usage': self.usage_frequency.mean().item(),
            'max_usage': self.usage_frequency.max().item(),
            'total_replacements': self.total_replacements.item(),
            'avg_replacements_per_slot': self.replacement_count.mean().item(),
            'most_replaced_slot': self.replacement_count.argmax().item(),
            'least_replaced_slot': self.replacement_count.argmin().item(),
            'total_forward_calls': self.total_forward_calls.item(),
            'skipped_updates': self.skipped_updates.item(),
        }
    
    def get_performance_stats(self) -> dict:
        """Get detailed performance statistics."""
        total_calls = max(self.total_forward_calls.item(), 1)
        actual_updates = max(total_calls - self.skipped_updates.item(), 1)
        
        return {
            'update_efficiency': {
                'total_forward_calls': total_calls,
                'skipped_updates': self.skipped_updates.item(),
                'actual_updates': actual_updates,
                'skip_ratio': self.skipped_updates.item() / total_calls,
                'update_frequency_setting': self.update_frequency,
            },
            'memory_efficiency': {
                'memory_utilization': (self.usage_frequency > 0).float().mean().item(),
                'active_slots': (self.usage_frequency > 0).sum().item(),
                'total_slots': self.num_slots,
                'avg_usage_per_slot': self.usage_frequency.mean().item(),
            },
            'replacement_efficiency': {
                'total_replacements': self.total_replacements.item(),
                'replacements_per_call': self.total_replacements.item() / total_calls,
                'most_active_slot_replacements': self.replacement_count.max().item(),
                'least_active_slot_replacements': self.replacement_count.min().item(),
            },
            'settings': {
                'enabled': self.enable_dynamic_update,
                'update_rate': self.update_rate,
                'novelty_threshold': self.novelty_threshold,
            }
        }


class TimeAttentionEmbeddings(nn.Module):
    """Time-based attention mechanism that mimics human memory decay and periodic patterns.
    
    Features:
    1. Temporal decay - exponential decay function to give higher weights to recent events
    2. Periodic attention - separate attention branches for different periodic patterns
    3. Feature-time fusion support - adaptively combines with features
    """
    def __init__(
        self,
        t_mean: float,
        t_std: float,
        d_embedding: int = 128,
        decay_factor: float = 0.1,
        periodic_patterns: Optional[List[float]] = None,
        feature_fusion: bool = True,
        num_harmonics: int = 1,  # Number of harmonics for periodic embeddings
        learnable_phase: bool = False,  # Whether to learn phase shifts
        **kwargs,
    ) -> None:
        super().__init__()
        self.out_dim = d_embedding
        self.t_mean = t_mean
        self.t_std = t_std
        self.decay_factor = decay_factor
        self.feature_fusion = feature_fusion
        self.num_harmonics = num_harmonics
        self.learnable_phase = learnable_phase
        
        # Default periodic patterns if none provided (e.g. day, week)
        self.periodic_patterns = periodic_patterns or [24.0, 24.0*7]  # hours in day, hours in week
        
        # Projection layers for different temporal aspects
        # Ensure all components have EXACTLY equal dimensions for weighted sum compatibility
        if feature_fusion:
            # All three components use the same dimension for weighted sum
            component_dim = d_embedding // 3
            self.decay_dim = component_dim
            self.period_dim = component_dim  
            self.context_dim = component_dim
        else:
            # Split between decay and period components only
            component_dim = d_embedding // 2
            self.decay_dim = component_dim
            self.period_dim = component_dim
            self.context_dim = 0
            
        # Store component dimension for router compatibility
        self.component_dim = component_dim
        
        # Update out_dim to reflect actual output dimensions
        if feature_fusion:
            self.out_dim = component_dim * 3  # decay + period + context
        else:
            self.out_dim = component_dim * 2  # decay + period
            
        self.decay_proj = nn.Linear(1, self.decay_dim)
        
        # Create individual periodic projections, then combine to target dimension
        n_patterns = len(self.periodic_patterns)
        individual_period_dim = max(1, self.period_dim // max(1, n_patterns))
        
        if self.learnable_phase:
            self.phase_shifts = nn.Parameter(torch.zeros(n_patterns, self.num_harmonics))

        self.periodic_projs = nn.ModuleList([
            nn.Linear(2 * self.num_harmonics, individual_period_dim)  # sin and cos for each harmonic
            for _ in self.periodic_patterns
        ])
        
        # Final projection to combine all periodic patterns to target dimension
        total_period_features = individual_period_dim * n_patterns
        if total_period_features != self.period_dim:
            self.period_final_proj = nn.Linear(total_period_features, self.period_dim)
        else:
            self.period_final_proj = None
        
        # Contextual time representation for feature fusion
        if self.feature_fusion:
            self.context_proj = nn.Sequential(
                nn.Linear(1, self.context_dim),
                nn.Tanh()
            )
        
        # Attention mechanism to combine different temporal aspects
        self.attention = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim)
        )
        
    def forward(self, t):
        # Normalize timestamps
        t_norm = (t - self.t_mean) / (self.t_std + 1e-8)
        
        # 1. Temporal decay component - exponential decay based on recency
        # Assuming t is sorted with most recent time at the end
        # We compute relative time differences from the most recent time
        if len(t.shape) > 1:
            latest_t = t.max(dim=1, keepdim=True)[0]
        else:
            latest_t = t.max().reshape(1)
        
        time_diff = latest_t - t
        decay_weights = torch.exp(-self.decay_factor * time_diff)
        decay_emb = self.decay_proj(decay_weights.unsqueeze(-1))
        
        # 2. Periodic patterns component
        period_embs = []
        for i, period in enumerate(self.periodic_patterns):
            # Harmonics
            harmonics = torch.arange(1, self.num_harmonics + 1, device=t.device).view(1, -1)
            # Convert to radians for the given period
            theta = 2 * math.pi * t_norm.unsqueeze(-1) / period
            theta = theta * harmonics
            
            if self.learnable_phase:
                theta = theta + self.phase_shifts[i]
                
            # Get sin and cos for circular embedding
            sin_t = torch.sin(theta)
            cos_t = torch.cos(theta)
            # Combine and project
            periodic_input = torch.cat([sin_t.unsqueeze(-1), cos_t.unsqueeze(-1)], dim=-1)
            period_emb = self.periodic_projs[i](periodic_input)
            period_embs.append(period_emb)
        
        # Combine all periodic embeddings
        if period_embs:
            period_emb = torch.cat(period_embs, dim=-1)
            # Apply final projection if needed to match target dimension
            if self.period_final_proj is not None:
                period_emb = self.period_final_proj(period_emb)
        else:
            period_emb = torch.zeros((t.shape[0], self.period_dim), device=t.device)
        
        # 3. Create contextual time representation for feature fusion (if enabled)
        if self.feature_fusion:
            context_emb = self.context_proj(t_norm.unsqueeze(-1))
            # Combine all embeddings
            combined_emb = torch.cat([decay_emb, period_emb, context_emb], dim=-1)
        else:
            # Combine decay and periodic embeddings
            combined_emb = torch.cat([decay_emb, period_emb], dim=-1)
            
        # Apply attention mechanism
        attention_weights = torch.sigmoid(self.attention(combined_emb))
        return combined_emb * attention_weights
    
    def forward_components(self, t):
        """Forward pass that returns separate components for pathway routing.
        
        Returns:
            tuple: (decay_emb, period_emb, context_emb) - three separate pathway embeddings
        """
        # Normalize timestamps
        t_norm = (t - self.t_mean) / (self.t_std + 1e-8)
        
        # 1. Temporal decay component - exponential decay based on recency
        if len(t.shape) > 1:
            latest_t = t.max(dim=1, keepdim=True)[0]
        else:
            latest_t = t.max().reshape(1)
        
        time_diff = latest_t - t
        decay_weights = torch.exp(-self.decay_factor * time_diff)
        decay_emb = self.decay_proj(decay_weights.unsqueeze(-1))
        
        # 2. Periodic patterns component
        period_embs = []
        for i, period in enumerate(self.periodic_patterns):
            # Harmonics
            harmonics = torch.arange(1, self.num_harmonics + 1, device=t.device).view(1, -1)
            # Convert to radians for the given period
            theta = 2 * math.pi * t_norm.unsqueeze(-1) / period
            theta = theta * harmonics
            
            if self.learnable_phase:
                theta = theta + self.phase_shifts[i]
                
            # Get sin and cos for circular embedding
            sin_t = torch.sin(theta)
            cos_t = torch.cos(theta)
            # Combine and project
            periodic_input = torch.cat([sin_t, cos_t], dim=-1)
            period_emb = self.periodic_projs[i](periodic_input)
            period_embs.append(period_emb)
        
        # Combine all periodic embeddings
        if period_embs:
            period_emb = torch.cat(period_embs, dim=-1)
            # Apply final projection if needed to match target dimension
            if self.period_final_proj is not None:
                period_emb = self.period_final_proj(period_emb)
        else:
            period_emb = torch.zeros((t.shape[0], self.period_dim), device=t.device)
        
        # 3. Create contextual time representation (if enabled)
        if self.feature_fusion:
            context_emb = self.context_proj(t_norm.unsqueeze(-1))
        else:
            # Create empty context embedding if feature fusion is disabled
            context_emb = torch.zeros((t.shape[0], self.context_dim), device=t.device)
            
        return decay_emb, period_emb, context_emb


class MemoryBankTimeAttentionEmbeddings(nn.Module):
    """Time-based attention mechanism with learnable memory bank for contextual representation.
    
    Features:
    1. Temporal decay - exponential decay function to give higher weights to recent events
    2. Periodic attention - separate attention branches for different periodic patterns
    3. Dynamic memory bank contextual representation - learnable memory slots with adaptive updates
    """
    def __init__(
        self,
        t_mean: float,
        t_std: float,
        d_embedding: int = 128,
        decay_factor: float = 0.1,
        periodic_patterns: Optional[List[float]] = None,
        feature_fusion: bool = True,
        # Memory bank parameters
        memory_slots: int = 16,
        distance_type: str = "l2",
        memory_temperature: float = 1.0,
        # Dynamic update parameters
        memory_update_rate: float = 0.1,
        memory_decay_factor: float = 0.99,
        min_activation_threshold: float = 0.1,
        use_memory_gating: bool = True,
        novelty_threshold: float = 0.7,
        cleanup_time_threshold: float = 100.0,
        cleanup_usage_threshold: float = 0.01,
        update_frequency: int = 10,
        enable_dynamic_update: bool = True,
        num_harmonics: int = 1,  # Number of harmonics for periodic embeddings
        learnable_phase: bool = False,  # Whether to learn phase shifts
        **kwargs,
    ) -> None:
        super().__init__()
        self.t_mean = t_mean
        self.t_std = t_std
        self.decay_factor = decay_factor
        self.feature_fusion = feature_fusion
        self.num_harmonics = num_harmonics
        self.learnable_phase = learnable_phase
        
        # Default periodic patterns if none provided (e.g. day, week)
        self.periodic_patterns = periodic_patterns or [24.0, 24.0*7]  # hours in day, hours in week
        
        # Projection layers for different temporal aspects
        # Ensure all components have EXACTLY equal dimensions for weighted sum compatibility
        if feature_fusion:
            # All three components use the same dimension for weighted sum
            component_dim = d_embedding // 3
            self.decay_dim = component_dim
            self.period_dim = component_dim  
            self.context_dim = component_dim
        else:
            # Split between decay and period components only
            component_dim = d_embedding // 2
            self.decay_dim = component_dim
            self.period_dim = component_dim
            self.context_dim = 0
            
        # Store component dimension for router compatibility
        self.component_dim = component_dim
        
        # Update out_dim to reflect actual output dimensions
        if feature_fusion:
            self.out_dim = component_dim * 3  # decay + period + context
        else:
            self.out_dim = component_dim * 2  # decay + period
            
        self.decay_proj = nn.Linear(1, self.decay_dim)
        
        # Create individual periodic projections, then combine to target dimension
        n_patterns = len(self.periodic_patterns)
        individual_period_dim = max(1, self.period_dim // max(1, n_patterns))
        
        if self.learnable_phase:
            self.phase_shifts = nn.Parameter(torch.zeros(n_patterns, self.num_harmonics))
            
        self.periodic_projs = nn.ModuleList([
            nn.Linear(2 * self.num_harmonics, individual_period_dim)  # sin and cos embeddings for each harmonic
            for _ in self.periodic_patterns
        ])
        
        # Final projection to combine all periodic patterns to target dimension
        total_period_features = individual_period_dim * n_patterns
        if total_period_features != self.period_dim:
            self.period_final_proj = nn.Linear(total_period_features, self.period_dim)
        else:
            self.period_final_proj = None
        
        # Dynamic memory bank for contextual time representation
        if self.feature_fusion:
            self.memory_bank = MemoryBank(
                num_slots=memory_slots,
                slot_dim=self.context_dim,
                distance_type=distance_type,
                temperature=memory_temperature,
                update_rate=memory_update_rate,
                decay_factor=memory_decay_factor,
                min_activation_threshold=min_activation_threshold,
                use_gating=use_memory_gating,
                novelty_threshold=novelty_threshold,
                cleanup_time_threshold=cleanup_time_threshold,
                cleanup_usage_threshold=cleanup_usage_threshold,
                update_frequency=update_frequency,
                enable_dynamic_update=enable_dynamic_update
            )
            
            # Input projection for memory updates
            self.memory_input_proj = nn.Sequential(
                nn.Linear(self.decay_dim + self.period_dim, self.context_dim),
                nn.ReLU()
            )
        
        # Attention mechanism to combine different temporal aspects
        self.attention = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim)
        )
        
    def forward(self, t, enable_memory_update: bool = True):
        # Normalize timestamps
        t_norm = (t - self.t_mean) / (self.t_std + 1e-8)
        
        # 1. Temporal decay component - exponential decay based on recency
        # Assuming t is sorted with most recent time at the end
        # We compute relative time differences from the most recent time
        if len(t.shape) > 1:
            latest_t = t.max(dim=1, keepdim=True)[0]
        else:
            latest_t = t.max().reshape(1)
        
        time_diff = latest_t - t
        decay_weights = torch.exp(-self.decay_factor * time_diff)
        decay_emb = self.decay_proj(decay_weights.unsqueeze(-1))
        
                # 2. Periodic patterns component
        period_embs = []
        for i, period in enumerate(self.periodic_patterns):
            # Harmonics
            harmonics = torch.arange(1, self.num_harmonics + 1, device=t.device).view(1, -1)
            # Convert to radians for the given period
            theta = 2 * math.pi * t_norm.unsqueeze(-1) / period
            theta = theta * harmonics
            
            if self.learnable_phase:
                theta = theta + self.phase_shifts[i]
                
            # Get sin and cos for circular embedding
            sin_t = torch.sin(theta)
            cos_t = torch.cos(theta)
            # Combine and project
            periodic_input = torch.cat([sin_t, cos_t], dim=-1)
            period_emb = self.periodic_projs[i](periodic_input)
            period_embs.append(period_emb)
        
        # Combine all periodic embeddings
        if period_embs:
            period_emb = torch.cat(period_embs, dim=-1)
            # Apply final projection if needed to match target dimension
            if self.period_final_proj is not None:
                period_emb = self.period_final_proj(period_emb)
        else:
            period_emb = torch.zeros((t.shape[0], self.period_dim), device=t.device)
        
        # 3. Dynamic memory bank contextual representation (if feature fusion enabled)
        if self.feature_fusion:
            # Prepare input for memory update
            update_input = None
            if enable_memory_update and self.training:
                # Combine decay and period embeddings as input to memory
                temporal_features = torch.cat([decay_emb, period_emb], dim=-1)
                update_input = self.memory_input_proj(temporal_features)
            
            # Query memory bank with potential update
            context_emb = self.memory_bank(t_norm, update_input)
            
            # Combine all embeddings
            combined_emb = torch.cat([decay_emb, period_emb, context_emb], dim=-1)
        else:
            # Combine decay and periodic embeddings
            combined_emb = torch.cat([decay_emb, period_emb], dim=-1)
            
        # Apply attention mechanism
        attention_weights = torch.sigmoid(self.attention(combined_emb))
        return combined_emb * attention_weights
    
    def forward_components(self, t, enable_memory_update: bool = True):
        """Forward pass that returns separate components for pathway routing.
        
        Returns:
            tuple: (decay_emb, period_emb, context_emb) - three separate pathway embeddings
        """
        # Normalize timestamps
        t_norm = (t - self.t_mean) / (self.t_std + 1e-8)
        
        # 1. Temporal decay component - exponential decay based on recency
        if len(t.shape) > 1:
            latest_t = t.max(dim=1, keepdim=True)[0]
        else:
            latest_t = t.max().reshape(1)
        
        time_diff = latest_t - t
        decay_weights = torch.exp(-self.decay_factor * time_diff)
        decay_emb = self.decay_proj(decay_weights.unsqueeze(-1))
        
        # 2. Periodic patterns component
        period_embs = []
        for i, period in enumerate(self.periodic_patterns):
            # Harmonics
            harmonics = torch.arange(1, self.num_harmonics + 1, device=t.device).view(1, -1)
            # Convert to radians for the given period
            theta = 2 * math.pi * t_norm.unsqueeze(-1) / period
            theta = theta * harmonics
            
            if self.learnable_phase:
                theta = theta + self.phase_shifts[i]
                
            # Get sin and cos for circular embedding
            sin_t = torch.sin(theta)
            cos_t = torch.cos(theta)
            # Combine and project
            periodic_input = torch.cat([sin_t, cos_t], dim=-1)
            period_emb = self.periodic_projs[i](periodic_input)
            period_embs.append(period_emb)
        
        # Combine all periodic embeddings
        if period_embs:
            period_emb = torch.cat(period_embs, dim=-1)
            # Apply final projection if needed to match target dimension
            if self.period_final_proj is not None:
                period_emb = self.period_final_proj(period_emb)
        else:
            period_emb = torch.zeros((t.shape[0], self.period_dim), device=t.device)
        
        # 3. Dynamic memory bank contextual representation (if enabled)
        if self.feature_fusion:
            # Prepare input for memory update
            update_input = None
            if enable_memory_update and self.training:
                # Combine decay and period embeddings as input to memory
                temporal_features = torch.cat([decay_emb, period_emb], dim=-1)
                update_input = self.memory_input_proj(temporal_features)
            
            # Query memory bank with potential update
            context_emb = self.memory_bank(t_norm, update_input)
        else:
            # Create empty context embedding if feature fusion is disabled
            context_emb = torch.zeros((t.shape[0], self.context_dim), device=t.device)
            
        return decay_emb, period_emb, context_emb
    
    def reset_memory(self):
        """Reset the dynamic memory bank."""
        if self.feature_fusion and hasattr(self.memory_bank, 'reset_memory'):
            self.memory_bank.reset_memory()
    
    def get_memory_stats(self) -> dict:
        """Get memory bank statistics."""
        if self.feature_fusion and hasattr(self.memory_bank, 'get_memory_stats'):
            return self.memory_bank.get_memory_stats()
        return {}
    
    def set_memory_update_rate(self, rate: float):
        """Dynamically adjust memory update rate."""
        if self.feature_fusion:
            self.memory_bank.update_rate = rate
    
    def enable_memory_updates(self, enabled: bool = True):
        """Enable or disable dynamic memory updates for performance."""
        if self.feature_fusion:
            self.memory_bank.enable_dynamic_update = enabled
    
    def set_update_frequency(self, frequency: int):
        """Set how often memory updates occur (every N steps)."""
        if self.feature_fusion:
            self.memory_bank.update_frequency = frequency
    
    def get_performance_stats(self) -> dict:
        """Get detailed performance statistics."""
        if self.feature_fusion and hasattr(self.memory_bank, 'get_performance_stats'):
            return self.memory_bank.get_performance_stats()
        return {}


def get_optimized_config(mode: str = "balanced") -> dict:
    """Get optimized configuration for different performance requirements.
    
    Args:
        mode: One of "training", "balanced", "fast_inference", "memory_saving"
    
    Returns:
        Optimized configuration dictionary
    """
    base_config = {
        "embedding_type": "memory_attention",
        "d_embedding": 64,
        "memory_slots": 16,
    }
    
    if mode == "training":
        # Full dynamic updates for best learning
        config = {
            **base_config,
            "update_frequency": 1,  # Update every step
            "enable_dynamic_update": True,
            "use_memory_gating": True,
            "memory_update_rate": 0.1,
        }
    elif mode == "balanced":
        # Good balance of performance and learning
        config = {
            **base_config,
            "update_frequency": 10,  # Update every 10 steps
            "enable_dynamic_update": True,
            "use_memory_gating": False,  # Simpler updates
            "memory_update_rate": 0.05,
        }
    elif mode == "fast_inference":
        # Maximum speed for inference
        config = {
            **base_config,
            "enable_dynamic_update": False,  # No updates at all
            "use_memory_gating": False,
            "memory_slots": 8,  # Fewer slots
        }
    elif mode == "memory_saving":
        # Minimal memory usage
        config = {
            **base_config,
            "d_embedding": 32,  # Smaller embeddings
            "memory_slots": 8,  # Fewer slots
            "update_frequency": 50,  # Very infrequent updates
            "enable_dynamic_update": True,
            "use_memory_gating": False,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'training', 'balanced', 'fast_inference', 'memory_saving'")
    
    return config


def create_temporal_embeddings(t_mean: float, t_std: float, config: dict) -> nn.Module:
    """Factory function to create temporal embeddings based on configuration."""
    embedding_type = config.get("embedding_type", "fourier")
    
    if embedding_type == "snn":
        return SpikingTemporalEmbeddings(
            t_mean=t_mean,
            t_std=t_std,
            n_inputs=config.get("n_inputs", 10),
            n_hidden=config.get("n_hidden", 20),
            d_embedding=config.get("d_embedding", 64),
            simulation_steps=config.get("simulation_steps", 5),
            tau_mem=config.get("tau_mem", 10.0),
            threshold=config.get("threshold", 1.0),
            trend=config.get("trend", True)
        )
    elif embedding_type == "attention":
        return TimeAttentionEmbeddings(
            t_mean=t_mean,
            t_std=t_std,
            d_embedding=config.get("d_embedding", 64),
            decay_factor=config.get("decay_factor", 0.1),
            periodic_patterns=config.get("periodic_patterns", [24.0, 24.0*7]),
            feature_fusion=config.get("feature_fusion", True),
            num_harmonics=config.get("num_harmonics", 1),
            learnable_phase=config.get("learnable_phase", False)
        )
    elif embedding_type == "memory_attention":
        return MemoryBankTimeAttentionEmbeddings(
            t_mean=t_mean,
            t_std=t_std,
            d_embedding=config.get("d_embedding", 64),
            decay_factor=config.get("decay_factor", 0.1),
            periodic_patterns=config.get("periodic_patterns", [24.0, 24.0*7]),
            feature_fusion=config.get("feature_fusion", True),
            memory_slots=config.get("memory_slots", 16),
            distance_type=config.get("distance_type", "l2"),
            memory_temperature=config.get("memory_temperature", 1.0),
            memory_update_rate=config.get("memory_update_rate", 0.1),
            memory_decay_factor=config.get("memory_decay_factor", 0.99),
            min_activation_threshold=config.get("min_activation_threshold", 0.1),
            use_memory_gating=config.get("use_memory_gating", True),
            novelty_threshold=config.get("novelty_threshold", 0.7),
            cleanup_time_threshold=config.get("cleanup_time_threshold", 100.0),
            cleanup_usage_threshold=config.get("cleanup_usage_threshold", 0.01),
            update_frequency=config.get("update_frequency", 10),
            enable_dynamic_update=config.get("enable_dynamic_update", True),
            num_harmonics=config.get("num_harmonics", 1),
            learnable_phase=config.get("learnable_phase", False)
        )
    else:  # Default to Fourier embeddings
        return TemporalEmbeddings(
            t_mean=t_mean,
            t_std=t_std,
            order=config.get("order", [1, 1, 1, 1]),
            trend=config.get("trend", True),
            d_embedding=config.get("d_embedding", 64)
        )
