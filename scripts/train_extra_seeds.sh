#!/usr/bin/env bash
# Train the extra model seeds that notebook 09's S11 error bars need.
#
#   ./scripts/train_extra_seeds.sh            # seeds 1-4 for mlp_digit and cifar10_cnn
#   ./scripts/train_extra_seeds.sh cnn        # CNN only (needs a GPU; ~150 epochs each)
#   ./scripts/train_extra_seeds.sh mlp        # MLP only (CPU, a couple of minutes)
#
# mnist_even_odd_mlp_8_4_0134 already has seeds 0-4, so it is not covered here.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
WHICH="${1:-all}"
SEEDS="${SEEDS:-1 2 3 4}"

if [ "$WHICH" = "all" ] || [ "$WHICH" = "mlp" ]; then
  for S in $SEEDS; do
    echo "=== mnist_digit_mlp_40_20 seed $S"
    "$PY" "$REPO/scripts/train.py" --arch SimpleMLP --hidden-dims 40 20 --output-dim 10 \
      --dataset MNIST --data-root "$REPO/data/" --label-transform identity \
      --epochs 5 --batch-size 64 --seed "$S" --exp-name mnist_digit_mlp_40_20 \
      --description "SimpleMLP 784->40->20->10 on MNIST digit classification, seed $S"
  done
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "cnn" ]; then
  # GPU job. 150 epochs matches the seed-0 checkpoint (~90% CIFAR-10 test accuracy);
  # drop --epochs if you only need seeds that are trained comparably, not identically.
  for S in $SEEDS; do
    echo "=== cifar10_cnn seed $S"
    "$PY" "$REPO/scripts/train.py" --arch SmallCNN --dataset CIFAR10 \
      --data-root "$REPO/data/" --label-transform identity \
      --channels 32 64 128 256 --n-classes 10 \
      --epochs 150 --batch-size 128 --seed "$S" --exp-name cifar10_cnn \
      --description "SmallCNN on CIFAR-10, seed $S"
  done
fi

echo "checkpoints now in $REPO/data/models:"
ls -1 "$REPO/data/models"
