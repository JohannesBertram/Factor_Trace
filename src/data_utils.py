import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor

from .training import correct


def _get_targets(dataset):
    """Return targets as a numpy array, unwrapping Subset if necessary."""
    if isinstance(dataset, Subset):
        t = dataset.dataset.targets
        t = t.numpy() if hasattr(t, 'numpy') else np.array(t)
        return t[np.array(dataset.indices)]
    t = dataset.targets
    return t.numpy() if hasattr(t, 'numpy') else np.array(t)


def get_mnist_loaders(batch_size=32, root='./data/', digit_filter=None):
    train_ds = datasets.MNIST(root, train=True, download=True, transform=ToTensor())
    test_ds = datasets.MNIST(root, train=False, download=True, transform=ToTensor())
    if digit_filter is not None:
        digit_set = set(digit_filter)
        def _keep(ds):
            idx = [i for i, (_, y) in enumerate(ds) if int(y) in digit_set]
            return Subset(ds, idx)
        train_ds = _keep(train_ds)
        test_ds = _keep(test_ds)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class _LabelTransformedDataset(Dataset):
    """Wraps a dataset, mapping each yielded label through `transform`."""

    def __init__(self, base, transform):
        self.base, self.transform = base, transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, int(self.transform(torch.as_tensor(y)))


def label_transformed_loader(loader, label_transform, **dl_kwargs):
    """Return a DataLoader identical to `loader` but with labels mapped through
    `label_transform`.

    Useful for BFT primary mode (`bft(model, loader, validate=True)`) when the
    model was trained on transformed labels (e.g. even/odd) while the underlying
    dataset yields raw labels. Returns `loader` unchanged when `label_transform`
    is None.
    """
    if label_transform is None:
        return loader
    ds = _LabelTransformedDataset(loader.dataset, label_transform)
    return DataLoader(ds, batch_size=loader.batch_size, shuffle=False, **dl_kwargs)


def collect_layer_inputs_generic(
    model,
    dataset_or_loader,
    *,
    layer_filter=None,
    label_transform=None,
    only_correct=True,
    n_per_class=None,
    device=None,
    batch_size=256,
):
    """Hook-based layer input collection for any nn.Module.

    Captures inputs to all layers that pass layer_filter, using forward pre-hooks.
    Works with SimpleMLP, CNNs, and any model that does not require special
    calling conventions.  Does not require model.linear_layer_indices().

    Parameters
    ----------
    model             : nn.Module
    dataset_or_loader : Dataset or DataLoader
    layer_filter      : callable(module) -> bool, or None
                        Selects which layers to capture inputs for.
                        Default: captures inputs to all nn.Linear layers.
    label_transform   : callable or None
    only_correct      : bool — keep only samples where argmax(output) == target.
                        Set False to collect all samples (e.g. OOD evaluation).
    n_per_class       : int or None — max samples per transformed class.
                        Only applied when only_correct=True; ignored otherwise.
    device            : torch device; defaults to model's first parameter device
    batch_size        : int — DataLoader batch size when dataset_or_loader is a Dataset

    Returns
    -------
    dict with keys:
        images        : (N, ...) float32 ndarray
        targets       : (N,) transformed labels
        digits        : (N,) original dataset labels
        preds         : (N,) predicted class indices
        layer_inputs  : list[ndarray] — (N, n_in) per matched layer, forward order
    """
    if device is None:
        device = next(model.parameters()).device
    if layer_filter is None:
        layer_filter = lambda m: isinstance(m, nn.Linear)

    named = [(name, mod) for name, mod in model.named_modules() if layer_filter(mod)]
    store = {name: [] for name, _ in named}

    def make_hook(name):
        def h(mod, inp):
            store[name].append(inp[0].detach().cpu())
        return h

    hooks = [mod.register_forward_pre_hook(make_hook(name)) for name, mod in named]

    if isinstance(dataset_or_loader, DataLoader):
        loader = dataset_or_loader
    else:
        loader = DataLoader(dataset_or_loader, batch_size=batch_size, shuffle=False)

    all_imgs, all_digits, all_preds_list = [], [], []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            preds = out.argmax(1).cpu().numpy()
            all_imgs.append(x.cpu().numpy())
            all_digits.append(y.numpy())
            all_preds_list.append(preds)

    for h in hooks:
        h.remove()

    imgs   = np.concatenate(all_imgs)
    digits = np.concatenate(all_digits)
    preds  = np.concatenate(all_preds_list)

    if label_transform is not None:
        targets = label_transform(torch.tensor(digits)).numpy()
    else:
        targets = digits.copy()

    keep = np.arange(len(imgs))
    if only_correct:
        keep = keep[preds[keep] == targets[keep]]
        if n_per_class is not None:
            per_class = []
            for cl in np.unique(targets[keep]):
                cl_idx = keep[targets[keep] == cl][:n_per_class]
                per_class.append(cl_idx)
            keep = np.sort(np.concatenate(per_class))

    layer_inputs = []
    for name, _ in named:
        arr = torch.cat(store[name]).numpy()
        layer_inputs.append(arr.reshape(len(arr), -1)[keep])

    return {
        'images':       imgs[keep],
        'targets':      targets[keep],
        'digits':       digits[keep],
        'preds':        preds[keep],
        'layer_inputs': layer_inputs,
    }


def get_imagenet_loaders(batch_size=128, root='./data/imagenet/', augment='baseline',
                         img_size=224, num_workers=4):
    """Return (train_loader, val_loader) for ImageNet.

    Expects ImageNet at root/ with train/ and val/ subdirectories (ImageFolder layout).
    Falls back to torchvision.datasets.ImageNet if the ILSVRC directory structure exists.

    augment: 'baseline' — RandomResizedCrop + RandomHorizontalFlip
             'strong'   — baseline + ColorJitter
             'none'     — no training augmentation (val transforms only)
    Val set always uses Resize(256) + CenterCrop + Normalize.
    """
    import torchvision.transforms as T
    from torchvision import datasets as tv_datasets

    _mean = (0.485, 0.456, 0.406)
    _std  = (0.229, 0.224, 0.225)

    test_tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(_mean, _std),
    ])

    if augment == 'baseline':
        train_tfm = T.Compose([
            T.RandomResizedCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(_mean, _std),
        ])
    elif augment == 'strong':
        train_tfm = T.Compose([
            T.RandomResizedCrop(img_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
            T.Normalize(_mean, _std),
        ])
    elif augment == 'none':
        train_tfm = test_tfm
    else:
        raise ValueError(f"augment must be 'baseline', 'strong', or 'none'; got {augment!r}")

    def _load(split, transform):
        try:
            return tv_datasets.ImageNet(root, split=split, transform=transform)
        except Exception:
            folder = 'train' if split == 'train' else 'val'
            return tv_datasets.ImageFolder(
                os.path.join(root, folder), transform=transform)

    train_ds = _load('train', train_tfm)
    val_ds   = _load('val',   test_tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256,        shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def get_cifar10_loaders(batch_size=128, root='./data/', augment='baseline'):
    """Return (train_loader, test_loader) for CIFAR-10.

    augment: 'baseline' — RandomCrop + HorizontalFlip
             'cutout'   — baseline + CutOut(8×8 patch)
             'strong'   — baseline + ColorJitter + CutOut(16×16 patch)
             'none'     — no training augmentation (ToTensor + Normalize only)
    Test set always uses ToTensor + Normalize only.
    """
    import torchvision.transforms as T
    from torchvision import datasets as tv_datasets

    _mean = (0.4914, 0.4822, 0.4465)
    _std  = (0.2470, 0.2435, 0.2616)

    class _CutOut:
        """Zero out a random square patch after normalisation."""
        def __init__(self, size):
            self.size = size

        def __call__(self, img):
            _, h, w = img.shape
            cy = torch.randint(0, h, (1,)).item()
            cx = torch.randint(0, w, (1,)).item()
            y1 = max(0, cy - self.size // 2)
            y2 = min(h, cy + self.size // 2)
            x1 = max(0, cx - self.size // 2)
            x2 = min(w, cx + self.size // 2)
            img = img.clone()
            img[:, y1:y2, x1:x2] = 0.0
            return img

    base = [T.ToTensor(), T.Normalize(_mean, _std)]

    if augment == 'baseline':
        train_tfm = T.Compose([T.RandomCrop(32, padding=4),
                               T.RandomHorizontalFlip()] + base)
    elif augment == 'cutout':
        train_tfm = T.Compose([T.RandomCrop(32, padding=4),
                               T.RandomHorizontalFlip()] + base + [_CutOut(8)])
    elif augment == 'strong':
        train_tfm = T.Compose([T.RandomCrop(32, padding=4),
                               T.RandomHorizontalFlip(),
                               T.ColorJitter(0.4, 0.4, 0.4, 0.1)] + base + [_CutOut(16)])
    elif augment == 'none':
        train_tfm = T.Compose(base)
    else:
        raise ValueError(f"augment must be 'baseline', 'cutout', 'strong', or 'none'; got {augment!r}")

    test_tfm = T.Compose(base)

    train_ds = tv_datasets.CIFAR10(root, train=True,  download=True, transform=train_tfm)
    test_ds  = tv_datasets.CIFAR10(root, train=False, download=True, transform=test_tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=256,         shuffle=False)
    return train_loader, test_loader


def imdenorm(img, mean, std):
    """Reverse normalisation: (C,H,W) float → (H,W,C) float clipped to [0,1].

    Parameters
    ----------
    img  : np.ndarray, shape (C, H, W)
    mean : sequence of C floats — per-channel mean used during normalisation
    std  : sequence of C floats — per-channel std used during normalisation
    """
    m = np.array(mean)[:, None, None]
    s = np.array(std)[:, None, None]
    return np.clip((img * s + m).transpose(1, 2, 0), 0.0, 1.0)
