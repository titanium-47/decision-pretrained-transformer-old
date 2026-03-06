#!/usr/bin/env bash
# Run train_context_accumulator.py with different kl_loss_weight values.
# Edit the KL_WEIGHTS array below to set which values to sweep.

# Configure kl_loss_weight values to sweep (one run per value)
ALPHA_VALUES=(0.8 0.9)

# Optional: pass through any extra args to the training script (e.g. --exp_name my_run)
# Either pass on the command line: ./run_kl_sweep.sh --log_wandb --wandb_project kl-sweep-brightroom
# Or add defaults here as separate array elements (not one string):
EXTRA_ARGS=("$@")

for alpha in "${ALPHA_VALUES[@]}"; do
  echo "=========================================="
  echo "Running with --alpha ${alpha}"
  echo "=========================================="
  python train_context_accumulator.py --exp_name maze-cnn-nogoal-nostate-"${alpha}" --log_wandb --wandb_project asteroid-maze --alpha "${alpha}" --state_encoder_type noop --encoder_type diffusion_forward_noise "${EXTRA_ARGS[@]}"
  echo ""
done

echo "Sweep finished."
