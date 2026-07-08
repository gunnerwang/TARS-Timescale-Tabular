import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Literal
from torch import Tensor
import delu
from model.lib.tabm.tabm import _init_scaling_by_sections
from model.lib.tabm.deep import ElementwiseAffineEnsemble, make_efficient_ensemble, OneHotEncoding0d
from model.lib.tabr.utils import make_module, make_module1, MLP # , ResNet
from model.models.tabm import _get_first_input_scaling
from model.lib.temporal_embeddings import TemporalEmbeddings, MemoryBankTimeAttentionEmbeddings
from model.models.mlp_temporal import MultiTimescaleImplicitEncoder, TemporalComponentRouter
    
DATASET_SIZE = 279415
# ct: 227087, de: 279415, mr: 160019, we: 340596;  hi: 224320, eo: 109341, hd: 267645, sh: 18847
BATCH_SIZE = 1024
    
class TabM_Temporal(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        t_mean: float,
        t_std: float,
        # bins: None | list[Tensor],
        backbone: dict,
        num_embeddings: None | dict = None,
        temporal_embeddings: None | dict = None,
        implicit_time_dim: int = 64,
        test_time_adaptation: bool = True,
        dataset_size: int = DATASET_SIZE,
        batch_size: int = BATCH_SIZE,
        arch_type: Literal[
            # Active
            'vanilla', # Simple MLP
            'tabm',  # BatchEnsemble + separate heads + better initialization
            'tabm-mini',  # Minimal: * weight
            # BatchEnsemble
            'tabm-naive'
        ],
        k: None | int = None,
    ) -> None:
        # >>> Validate arguments.
        assert n_num_features >= 0
        assert n_num_features or cat_cardinalities
        if arch_type == 'vanilla':
            assert k is None
        else:
            assert k is not None
            assert k > 0
        if cat_cardinalities is None:
            cat_cardinalities = []
        super().__init__()

        # >>> Continuous (numerical) features + advanced temporal mechanism
        scaling_init_sections = []
        self.implicit_time_dim = implicit_time_dim
        self.test_time_adaptation = test_time_adaptation
        self.temporal_router = None
        self.fusion_gate = None
        self.temporal_embeddings = None
        actual_temporal_dim = 0
        if temporal_embeddings is not None:
            base_config = temporal_embeddings.copy()
            unified_config = base_config.copy()
            unified_config.update({
                'decay_factor': base_config.get('decay_factor', 0.1),
                'periodic_patterns': base_config.get('periodic_patterns', [1, 24, 24*7]),
                'd_embedding': base_config.get('d_embedding', 64),
                'feature_fusion': base_config.get('feature_fusion', True),
            })
            self.temporal_embeddings = MemoryBankTimeAttentionEmbeddings(t_mean, t_std, **unified_config)
            if self.implicit_time_dim > 0:
                self.implicit_time_encoder = MultiTimescaleImplicitEncoder(
                    d_in=n_num_features,
                    d_out=self.implicit_time_dim,
                    dataset_size=dataset_size,
                    batch_size=batch_size,
                    test_time_adaptation=test_time_adaptation,
                )
                # If context_dim == 0 (feature_fusion disabled), skip router and use full temporal embedding
                if getattr(self.temporal_embeddings, 'context_dim', 0) and self.temporal_embeddings.context_dim > 0:
                    fast_dim = self.implicit_time_dim // 3
                    slow_dim = self.implicit_time_dim // 3
                    ultra_dim = self.implicit_time_dim - fast_dim - slow_dim
                    self.temporal_router = TemporalComponentRouter(
                        fast_dim=fast_dim,
                        slow_dim=slow_dim,
                        ultra_dim=ultra_dim,
                        fusion_strategy='softmax',
                        initial_temperature=5.0,
                        final_temperature=1.0,
                        temperature_schedule='linear',
                        dataset_size=dataset_size,
                        batch_size=batch_size,
                    )
                    actual_temporal_dim = self.temporal_embeddings.component_dim
                    self.fusion_gate = nn.Linear(actual_temporal_dim, n_num_features)
                else:
                    # No context pathway available; use unified embedding output dimension
                    actual_temporal_dim = self.temporal_embeddings.out_dim
                    self.temporal_router = None
                    self.fusion_gate = nn.Linear(actual_temporal_dim, n_num_features)
            else:
                actual_temporal_dim = self.temporal_embeddings.out_dim
        n_num_features_total = n_num_features + actual_temporal_dim

        if num_embeddings is None:
            self.num_module = None
            d_num = n_num_features_total
            scaling_init_sections.extend(1 for _ in range(n_num_features_total))

        else:
            self.num_module = make_module(
                num_embeddings, n_features=n_num_features_total
            )
            d_num = n_num_features_total * num_embeddings['d_embedding']
            scaling_init_sections.extend(
                num_embeddings['d_embedding'] for _ in range(n_num_features_total)
            )

        # >>> Categorical features
        self.cat_module = (
            OneHotEncoding0d(cat_cardinalities) if cat_cardinalities else None
        )
        scaling_init_sections.extend(cat_cardinalities)
        d_cat = sum(cat_cardinalities)

        # >>> Backbone
        d_flat = d_num + d_cat
        self.affine_ensemble = None
        self.backbone = make_module1(d_in=d_flat,**backbone)

        if arch_type != 'vanilla':
            assert k is not None
            scaling_init = (
                'random-signs'
                if num_embeddings is None
                else 'normal'
            )

            if arch_type == 'tabm-mini':
                # The minimal possible efficient ensemble.
                self.affine_ensemble = ElementwiseAffineEnsemble(
                    k,
                    d_flat,
                    weight=True,
                    bias=False,
                    weight_init=(
                        'random-signs'
                        if num_embeddings is None
                        else 'normal'
                    ),
                )
                _init_scaling_by_sections(
                    self.affine_ensemble.weight,  # type: ignore[code]
                    scaling_init,
                    scaling_init_sections,
                )

            elif arch_type == 'tabm-naive':
                # The original BatchEnsemble.
                make_efficient_ensemble(
                    self.backbone,
                    k=k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=True,
                    scaling_init='random-signs',
                )
            elif arch_type == 'tabm':
                # Like BatchEnsemble, but all scalings, except for the first one,
                # are initialized with ones.
                make_efficient_ensemble(
                    self.backbone,
                    k=k,
                    ensemble_scaling_in=True,
                    ensemble_scaling_out=True,
                    ensemble_bias=True,
                    scaling_init='ones',
                )
                _init_scaling_by_sections(
                    _get_first_input_scaling(self.backbone).r,  # type: ignore[code]
                    scaling_init,
                    scaling_init_sections,
                )

            else:
                raise ValueError(f'Unknown arch_type: {arch_type}')

        # >>> Output
        d_block = backbone['d_layers'][-1]
        self.d_out = 1 if n_classes is None else n_classes
        self.output = (
            nn.Linear(d_block, self.d_out)
            if arch_type == 'vanilla'
            else delu.nn.NLinear(k, d_block, self.d_out)  # type: ignore[code]
        )

        # >>>
        self.arch_type = arch_type
        self.k = k

    def forward(
        self, x_num: None | Tensor = None, x_cat: None | Tensor = None, idx: Tensor = None
    ) -> Tensor:
        x = []
        # Advanced temporal integration
        if self.temporal_embeddings is not None:
            if (
                self.temporal_router is not None
                and hasattr(self.temporal_embeddings, 'forward_components')
                and self.implicit_time_dim > 0
                and x_num is not None
            ):
                drift_encoding_dict = self.implicit_time_encoder(x_num)
                component_embeddings = self.temporal_embeddings.forward_components(idx)
                temporal_emb = self.temporal_router(drift_encoding_dict, component_embeddings)
            else:
                temporal_emb = self.temporal_embeddings(idx)
                temporal_emb = temporal_emb.flatten(1)

            if self.fusion_gate is not None and x_num is not None:
                temporal_weights = torch.sigmoid(self.fusion_gate(temporal_emb))
                x_weighted = x_num * temporal_weights
                x_num = x_weighted + x_num * 0.2

            if x_num is None:
                x_num = temporal_emb
            else:
                x_num = torch.cat([x_num, temporal_emb], dim=-1)
        if x_num is not None:
            x.append(x_num if self.num_module is None else self.num_module(x_num))
        if x_cat is None:
            assert self.cat_module is None
        else:
            assert self.cat_module is not None
            x.append(self.cat_module(x_cat))
        x = torch.column_stack([x_.flatten(1, -1) for x_ in x])
        if x.dtype == torch.int64:
            x = x.float()

        if self.k is not None:
            x = x[:, None].expand(-1, self.k, -1)  # (B, D) -> (B, K, D)
            if self.affine_ensemble is not None:
                x = self.affine_ensemble(x)
        else:
            assert self.affine_ensemble is None

        x = self.backbone(x)
        x = self.output(x)
        if self.k is None:
            x = x[:, None]
        if self.d_out == 1:
            x = x.squeeze(-1)
        return x