#!/usr/bin/env bash
# Run train_context_accumulator.py with different kl_loss_weight values.
# Edit the KL_WEIGHTS array below to set which values to sweep.

# Configure kl_loss_weight values to sweep (one run per value)
KL_WEIGHTS=(100.0)

# Optional: pass through any extra args to the training script (e.g. --exp_name my_run)
# Either pass on the command line: ./run_kl_sweep.sh --log_wandb --wandb_project kl-sweep-brightroom
# Or add defaults here as separate array elements (not one string):
EXTRA_ARGS=("$@")

for kl in "${KL_WEIGHTS[@]}"; do
  echo "=========================================="
  echo "Running with --kl_loss_weight ${kl}"
  echo "=========================================="
  python train_context_accumulator.py --log_wandb --wandb_project kl-sweep-brightroom --kl_loss_weight "${kl}" "${EXTRA_ARGS[@]}"
  echo ""
done

echo "Sweep finished."
