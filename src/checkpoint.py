"""
Experiment save/load utilities.

Every experiment lives in its own directory containing:
  weights.pt   — model.state_dict()
  config.json  — architecture + dataset + analysis parameters

config.json schema
------------------
{
  "arch":                 "SimpleMLP",
  "arch_kwargs":          {"input_dim": 784, "hidden_dims": [20, 10], "output_dim": 2},
  "dataset":              "MNIST",
  "dataset_kwargs":       {"root": "../data/", "batch_size": 32},
  "label_transform":      "even_odd",   # key into TRANSFORM_REGISTRY, or null
  "analysis_layer_indices": [2, 4, 6],  # feature_maps indices to concatenate
  "n_per_class":          1000,
  "input_side":           28,
  "description":          "human-readable experiment name"
}

Adding a new architecture
-------------------------
1. Implement an nn.Module subclass in src/models.py.
2. Add an entry to MODEL_REGISTRY below.

Adding a new dataset
--------------------
1. Write a loader function with signature (batch_size, root, **kwargs)
   -> (train_loader, test_loader).
2. Add an entry to DATASET_REGISTRY below.
"""

import json
import os
import torch

from .models import SimpleMLP, SmallCNN, TinyViT
from .training import label_transform_even_odd
from .data_utils import get_mnist_loaders, get_cifar10_loaders


MODEL_REGISTRY = {
    "SimpleMLP": SimpleMLP,
    "SmallCNN":  SmallCNN,
    "TinyViT":   TinyViT,
}

TRANSFORM_REGISTRY = {
    "even_odd": label_transform_even_odd,
    "identity": None,
}

DATASET_REGISTRY = {
    "MNIST":   get_mnist_loaders,
    "CIFAR10": get_cifar10_loaders,
}


def save_experiment(model, config, exp_dir):
    """
    Save model weights and config to exp_dir.

    Parameters
    ----------
    model   : nn.Module
    config  : dict matching the schema above
    exp_dir : str or path — created if it does not exist
    """
    os.makedirs(exp_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(exp_dir, "weights.pt"))
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_experiment(exp_dir, device=None):
    """
    Load a saved experiment.

    Returns (model, config) where model is on device and in eval mode.
    """
    if device is None:
        device = torch.device("cpu")
    with open(os.path.join(exp_dir, "config.json")) as f:
        config = json.load(f)

    arch_cls = MODEL_REGISTRY[config["arch"]]
    model = arch_cls(**config["arch_kwargs"]).to(device)
    model.load_state_dict(
        torch.load(os.path.join(exp_dir, "weights.pt"), map_location=device)
    )
    model.eval()
    return model, config


def get_transform(name):
    """Return the label-transform callable for name, or None for identity."""
    if name not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform '{name}'. Options: {list(TRANSFORM_REGISTRY)}")
    return TRANSFORM_REGISTRY[name]


def get_loaders_from_config(config):
    """Instantiate train/test DataLoaders from a config dict."""
    loader_fn = DATASET_REGISTRY[config["dataset"]]
    kwargs = dict(config.get("dataset_kwargs", {}))
    batch_size = kwargs.pop("batch_size", 32)
    return loader_fn(batch_size=batch_size, **kwargs)
