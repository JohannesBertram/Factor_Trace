import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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


def collect_activations(model, dataset, layer_indices, label_transform=None,
                        n_per_class=1000, device=None):
    """
    Collect intermediate layer activations for correctly classified samples.

    Parameters
    ----------
    model         : nn.Module with inference-mode forward(x, inference=True)
    dataset       : torchvision dataset
    layer_indices : list of int — which feature_maps indices to concatenate
    label_transform : callable(tensor) -> tensor, e.g. label_transform_even_odd
    n_per_class   : max samples per (transformed) class
    device        : torch device; defaults to model's device

    Returns
    -------
    dict with keys:
      'by_class'  : {cl: (n, n_neurons) ndarray}
      'all'       : (total_n, n_neurons) ndarray
      'targets'   : (total_n,) transformed labels
      'images'    : (total_n, ...) images as numpy
      'digits'    : (total_n,) original digit labels
      'classes'   : sorted list of unique transformed class ids
    """
    if device is None:
        device = next(model.parameters()).device

    raw_targets = _get_targets(dataset)

    if label_transform is not None:
        transformed = label_transform(torch.tensor(raw_targets)).numpy()
    else:
        transformed = raw_targets.copy()

    classes = sorted(np.unique(transformed).tolist())

    acts_by_class = {cl: [] for cl in classes}
    imgs_by_class = {cl: [] for cl in classes}
    digits_by_class = {cl: [] for cl in classes}

    model.eval()
    with torch.no_grad():
        for cl in classes:
            idxs = np.flatnonzero(transformed == cl)
            for i in idxs[:n_per_class]:
                data, digit = dataset[i]
                t_val = int(transformed[i])
                x = data.unsqueeze(0).to(device)
                output, feature_maps = model(x, inference=True)
                target_t = torch.tensor([t_val], device=device)
                if correct(output, target_t):
                    act = torch.cat([feature_maps[li].flatten() for li in layer_indices])
                    acts_by_class[cl].append(act.cpu().numpy())
                    imgs_by_class[cl].append(data.numpy())
                    digits_by_class[cl].append(int(digit))

    for cl in classes:
        acts_by_class[cl] = np.array(acts_by_class[cl])

    all_acts = np.concatenate([acts_by_class[cl] for cl in classes], axis=0)
    all_targets = np.concatenate([[cl] * len(acts_by_class[cl]) for cl in classes])
    all_images = np.concatenate([imgs_by_class[cl] for cl in classes], axis=0)
    all_digits = np.concatenate([digits_by_class[cl] for cl in classes])

    return {
        'by_class': acts_by_class,
        'all': all_acts,
        'targets': all_targets,
        'images': all_images,
        'digits': all_digits,
        'classes': classes,
    }



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


def collect_layer_inputs(model, dataset, label_transform=None, n_per_class=None, device=None):
    """
    Collect per-sample layer inputs and post-activation outputs for all linear layers.

    Only correctly classified samples are kept.  n_per_class=None collects all.

    Parameters
    ----------
    model           : SimpleMLP with linear_layer_indices()
    dataset         : torchvision dataset
    label_transform : callable or None
    n_per_class     : max samples per transformed class; None = all
    device          : torch device

    Returns
    -------
    dict with keys:
        images        : (n, C, H, W) raw images as float32 ndarray
        targets       : (n,) transformed class labels
        digits        : (n,) original dataset labels
        layer_inputs  : list[ndarray] — (n, n_in) per linear layer, first-to-last
                        (pixel inputs at index 0)
        layer_acts    : list[ndarray] — (n, n_out) post-activation per linear layer
    """
    if device is None:
        device = next(model.parameters()).device

    raw_targets = _get_targets(dataset)
    transformed = (label_transform(torch.tensor(raw_targets)).numpy()
                   if label_transform is not None else raw_targets.copy())
    classes = sorted(np.unique(transformed).tolist())

    linear_indices = model.linear_layer_indices()
    # Input to linear layer li lives at feature_maps[li - 1]; output at feature_maps[li + 1]
    input_fmap_idx = [li - 1 for li in linear_indices]
    act_fmap_idx = [li + 1 for li in linear_indices]

    imgs, tgts, digs = [], [], []
    layer_in_acc = [[] for _ in linear_indices]
    layer_act_acc = [[] for _ in linear_indices]

    model.eval()
    with torch.no_grad():
        for cl in classes:
            idxs = np.flatnonzero(transformed == cl)
            for i in idxs[:n_per_class]:
                data, digit = dataset[i]
                t_val = int(transformed[i])
                x = data.unsqueeze(0).to(device)
                output, feature_maps = model(x, inference=True)
                target_t = torch.tensor([t_val], device=device)
                if correct(output, target_t):
                    imgs.append(data.numpy())
                    tgts.append(t_val)
                    digs.append(int(digit))
                    for k, (ii, ai) in enumerate(zip(input_fmap_idx, act_fmap_idx)):
                        layer_in_acc[k].append(feature_maps[ii].cpu().numpy().flatten())
                        layer_act_acc[k].append(feature_maps[ai].cpu().numpy().flatten())

    return {
        'images': np.array(imgs),
        'targets': np.array(tgts),
        'digits': np.array(digs),
        'layer_inputs': [np.array(acc) for acc in layer_in_acc],
        'layer_acts': [np.array(acc) for acc in layer_act_acc],
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
