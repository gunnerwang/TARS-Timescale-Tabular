#!/usr/bin/env bash
# Example: train TARS (mlp_temporal) on cooking-time with tuning.
python train_model_deep.py --dataset cooking-time \
                           --model_type mlp_temporal \
                           --cat_policy ohe \
                           --enable_timestamp \
                           --gpu 0 --max_epoch 200 --seed_num 15 \
                           --validate_option holdout_foremost_sample \
                           --tune --retune --n_trials 100
