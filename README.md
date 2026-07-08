# TARS: Multi-Timescale Representation with Adaptive Routing for Deep Tabular Learning under Temporal Shift

Official code for the **Neural Networks** (2026) paper **[Multi-timescale representation with adaptive routing for deep tabular learning under temporal shift](https://doi.org/10.1016/j.neunet.2026.108670)** (TARS), by Tianyu Wang, Maite Zhang, Mingxuan Lu, and Mian Li.

TARS augments deep tabular backbones with (i) a **multi-timescale implicit temporal representation** and (ii) a **drift-adaptive routing** mechanism that bypasses drift-based routing when the temporal-shift signal is weak. Experiments follow the TabReD benchmark protocol of Cai & Ye (2025) [1].

## Setup

```bash
conda create --name tars python=3.10
conda activate tars
pip install -r requirements.txt
conda install faiss-gpu -c pytorch          # only needed for TabR / ModernNCA
```

## Data

Experiments use the [TabReD](https://github.com/yandex-research/tabred) benchmark [2]. Download the raw datasets from TabReD and place them under `tabred/data/` (see their instructions). The fixed random validation splits used by our protocol are already provided in `data_splits/`.

## Usage

### Deep methods

```bash
python train_model_deep.py --dataset $DATASET \
                           --model_type $MODEL \
                           --cat_policy $CAT_POLICY \
                           --enable_timestamp \
                           --gpu 0 --max_epoch 200 --seed_num 15 \
                           --validate_option holdout_foremost_sample \
                           --tune --retune --n_trials 100
```

- `$DATASET`: `cooking-time, delivery-eta, ecom-offers, homecredit-default, homesite-insurance, maps-routing, sberbank-housing, weather`
- `$MODEL`: `*_temporal` is the TARS variant of each backbone; plain names are the non-temporal baseline; `*_modulated` is the [Feature-aware Modulation](https://arxiv.org/abs/2512.03678) baseline.

  ```
  mlp,        mlp_temporal,
  mlp_plr,    mlp_plr_temporal,
  snn,        snn_temporal,
  dcn2,       dcn2_temporal,
  ftt,        ftt_temporal,
  tabr,       tabr_temporal,
  modernNCA,  modernNCA_temporal,
  tabm,       tabm_temporal,
  mlp_modulated, tabm_modulated,
  ```

- `$CAT_POLICY`:

  ```bash
  case $MODEL in
    modernNCA*|tabr*)              cat_policy=tabr_ohe ;;
    mlp_plr*|tabm*|ftt*|dcn2*|snn*) cat_policy=indices ;;
    *)                            cat_policy=ohe ;;
  esac
  ```

#### TARS knobs (on the `*_temporal` model_types)

These are tuned automatically under `--tune`, or can be fixed in `configs/`:

- `implicit_time_dim` — dimension of the multi-timescale implicit encoder (0 disables it; reproduces the plain temporal-embedding baseline of [1]).
- `enable_bypass` — enable drift-adaptive bypass routing.
- `test_time_adaptation` — adapt the temporal representation at test time as the timestamp advances.
- `drift_significance_threshold` — drift magnitude above which drift-based routing is used.
- `use_cbp`, `replacement_rate`, `maturity_threshold`, `cbp_init` — continual-backpropagation options.
- `temporal_embeddings.{d_embedding, decay_factor, periodic_patterns}` — multi-timescale embedding config.

### Classical methods

```bash
python train_model_classical.py --dataset $DATASET \
                                --model_type $MODEL \
                                --cat_policy $CAT_POLICY \
                                --enable_timestamp \
                                --gpu "" --seed_num 15 \
                                --validate_option holdout_foremost_sample \
                                --tune --retune --n_trials 100
```

- `$MODEL`: `XGBoost, LightGBM, CatBoost, RandomForest, SGD` (Linear). Pass as `xgboost, lightgbm, catboost, RandomForest, SGD`.
- `$CAT_POLICY`: `indices` for CatBoost, else `ohe`.

## Baselines

We compare against the following external methods (not bundled here — see their repositories):

- **AdapTable** — test-time adaptation for tabular data.
- **FTTA** — fully test-time adaptation (https://github.com/WNJXYK/FTTA).
- **Koodos** — Koopman-style temporal domain generalization.
- **TabPFN / Drift-Resilient TabPFN** — in-context-learning tabular foundation models.

## References

[1] Cai, H.-R. and Ye, H.-J. Understanding the Limits of Deep Tabular Methods with Temporal Shift. ICML, 2025. [arXiv:2502.20260](https://arxiv.org/abs/2502.20260)

[2] Rubachev, I., Kartashev, N., Gorishniy, Y., and Babenko, A. TabReD: A Benchmark of Tabular Machine Learning In-the-Wild. ICLR, 2025.

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2026tars,
  title   = {Multi-timescale representation with adaptive routing for deep tabular learning under temporal shift},
  author  = {Wang, Tianyu and Zhang, Maite and Lu, Mingxuan and Li, Mian},
  journal = {Neural Networks},
  volume  = {199},
  pages   = {108670},
  year    = {2026},
  doi     = {10.1016/j.neunet.2026.108670}
}
```

**Enjoy the code!**
