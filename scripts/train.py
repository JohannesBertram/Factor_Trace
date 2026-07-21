"""
Standalone training script. Trains a model and saves it to
data/models/<exp_name>_seed<N>/ (where the notebooks look for checkpoints).

Usage
-----
python scripts/train.py --exp-name mnist_even_odd_mlp_20_10

Extra seeds for the notebook-09 error bars (S11):
python scripts/train.py --arch SimpleMLP --hidden-dims 40 20 --output-dim 10 \
  --dataset MNIST --label-transform identity --seed 1 --exp-name mnist_digit_mlp_40_20
python scripts/train.py --arch SmallCNN --dataset CIFAR10 --label-transform identity \
  --channels 32 64 128 256 --n-classes 10 --epochs 150 --batch-size 128 \
  --seed 1 --exp-name cifar10_cnn

Full example with overrides:
python scripts/train.py \
  --arch SimpleMLP \
  --hidden-dims 50 20 10 \
  --output-dim 2 \
  --dataset MNIST \
  --data-root ../data/ \
  --label-transform even_odd \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --n-per-class 1000 \
  --analysis-layers 2 4 6 \
  --input-side 28 \
  --exp-name mnist_even_odd_mlp_50_20_10 \
  --description "MNIST even/odd — MLP 784→50→20→10→2"
"""

import argparse
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.checkpoint import (
    MODEL_REGISTRY, TRANSFORM_REGISTRY, DATASET_REGISTRY,
    save_experiment, get_transform,
)
from src.training import train_epoch, evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Train a model and save to experiments/")
    p.add_argument("--arch", default="SimpleMLP", choices=list(MODEL_REGISTRY))
    p.add_argument("--hidden-dims", nargs="+", type=int, default=[20, 10])
    p.add_argument("--output-dim", type=int, default=2)
    p.add_argument("--dataset", default="MNIST", choices=list(DATASET_REGISTRY))
    p.add_argument("--data-root", default="../data/")
    p.add_argument("--label-transform", default="even_odd", choices=list(TRANSFORM_REGISTRY))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-per-class", type=int, default=1000)
    p.add_argument("--analysis-layers", nargs="+", type=int, default=[2, 4, 6],
                   help="feature_maps indices to concatenate for analysis")
    p.add_argument("--input-side", type=int, default=28)
    p.add_argument("--exp-name", required=True,
                   help="subdirectory name under --out-root; '_seed<N>' is appended "
                        "automatically unless already present")
    p.add_argument("--out-root", default=None,
                   help="where to save (default: data/models/, which is where the "
                        "notebooks look for checkpoints)")
    p.add_argument("--description", default="")
    p.add_argument("--digit-filter", nargs="+", type=int, default=None,
                   help="Restrict data to these digit labels (e.g. --digit-filter 0 1 3 4)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for torch and numpy")
    # SmallCNN
    p.add_argument("--channels", nargs="+", type=int, default=[32, 64, 128, 256])
    p.add_argument("--fc-dim", type=int, default=128)
    p.add_argument("--n-classes", type=int, default=10)
    p.add_argument("--no-global-pool", action="store_true")
    # TinyViT
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--ffn-dim", type=int, default=64)
    return p.parse_args()


def build_arch_kwargs(args, input_dim):
    """Per-architecture constructor kwargs.

    SimpleMLP is input_dim/hidden_dims/output_dim; SmallCNN and TinyViT take entirely
    different signatures, so hardcoding the MLP triple made them untrainable here.
    """
    if args.arch == "SmallCNN":
        return dict(channels=tuple(args.channels), fc_dim=args.fc_dim,
                    n_classes=args.n_classes, global_pool=not args.no_global_pool)
    if args.arch == "TinyViT":
        return dict(embed_dim=args.embed_dim, n_heads=args.n_heads,
                    ffn_dim=args.ffn_dim, n_classes=args.n_classes)
    return dict(input_dim=input_dim, hidden_dims=args.hidden_dims,
                output_dim=args.output_dim)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # --- data ---
    loader_fn = DATASET_REGISTRY[args.dataset]
    loader_kwargs = {'batch_size': args.batch_size, 'root': args.data_root}
    if args.digit_filter:
        loader_kwargs['digit_filter'] = args.digit_filter
    train_loader, test_loader = loader_fn(**loader_kwargs)

    label_transform = get_transform(args.label_transform)

    # --- model ---
    arch_cls = MODEL_REGISTRY[args.arch]
    # infer input_dim from first batch
    sample, _ = next(iter(train_loader))
    input_dim = sample[0].numel()

    arch_kwargs = build_arch_kwargs(args, input_dim)
    model = arch_cls(**arch_kwargs).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- train ---
    for epoch in range(args.epochs):
        loss, acc = train_epoch(model, train_loader, optimizer, criterion,
                                device, label_transform=label_transform)
        print(f"epoch {epoch+1}  loss={loss:.4f}  acc={acc:.2%}")

    test_loss, test_acc = evaluate(model, test_loader, criterion,
                                   device, label_transform=label_transform)
    print(f"\ntest  loss={test_loss:.4f}  acc={test_acc:.2%}")

    # --- save ---
    config = {
        "arch": args.arch,
        "arch_kwargs": arch_kwargs,
        "dataset": args.dataset,
        "dataset_kwargs": loader_kwargs,
        "label_transform": args.label_transform,
        "analysis_layer_indices": args.analysis_layers,
        "n_per_class": args.n_per_class,
        "input_side": args.input_side,
        "seed": args.seed,
        "description": args.description or f"{args.arch} on {args.dataset} ({args.label_transform})",
    }

    # The notebooks look under data/models/<name>_seed<N>, so default there and append the
    # seed suffix unless the caller already spelled it out.
    name = args.exp_name
    if not re.search(r"_seed\d+$", name):
        name = f"{name}_seed{args.seed}"
    out_root = args.out_root or os.path.join(os.path.dirname(__file__), "..", "data", "models")
    exp_dir = os.path.join(out_root, name)
    save_experiment(model, config, exp_dir)
    print(f"\nSaved to {os.path.abspath(exp_dir)}")


if __name__ == "__main__":
    main()
