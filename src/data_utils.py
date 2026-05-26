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


def collect_layer_data(model, loader, device, only_correct=True):
    """Hook-based collection of activations for every Conv2d and Linear layer.

    Registers forward hooks on all Conv2d and Linear sub-modules (in the order
    returned by model.named_modules()), runs the dataloader, and assembles
    per-sample inputs/outputs for each layer.

    Parameters
    ----------
    model        : nn.Module — any architecture with Conv2d / Linear layers
    loader       : DataLoader — yields (images, labels) batches
    device       : torch device
    only_correct : bool — when True, keeps only samples where argmax(output)==label

    Returns
    -------
    dict with keys:
        'images'      : (N, C, H, W) float32 numpy array
        'targets'     : (N,) int numpy array of ground-truth labels
        'confidences' : (N,) float32 numpy array — max output probability per sample
        'layer_data'  : list of dicts, one per Conv2d/Linear in forward order:
            {'name': str, 'type': 'conv'|'fc',
             'weight':      ndarray,  # detached, shape matches layer convention
             'input_fmap':  ndarray,  # (N, ...) matching layer input shape
             'output_fmap': ndarray}  # (N, ...) matching layer output shape
    """
    model.eval()
    named = [(n, m) for n, m in model.named_modules()
             if isinstance(m, (nn.Conv2d, nn.Linear))]
    # Per-batch hook storage; each hook overwrites the slot for its layer.
    store = {n: {'inp': None, 'out': None} for n, _ in named}

    def make_hook(name):
        def h(mod, inp, out):
            store[name]['inp'] = inp[0].detach().cpu()
            store[name]['out'] = out.detach().cpu()
        return h

    hooks = [m.register_forward_hook(make_hook(n)) for n, m in named]
    acc_inp = {n: [] for n, _ in named}
    acc_out = {n: [] for n, _ in named}
    all_imgs, all_tgts, all_confs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            # Support log-softmax output (negative values) as well as raw probs.
            probs = out.exp() if out.min() < 0 else out
            confs = probs.max(1).values
            preds = probs.argmax(1)
            ok = (preds == y).cpu().nonzero(as_tuple=True)[0] if only_correct \
                 else torch.arange(len(y))
            if not len(ok):
                continue
            all_imgs.append(x[ok].cpu())
            all_tgts.append(y[ok].cpu())
            all_confs.append(confs[ok].cpu())
            for n, _ in named:
                acc_inp[n].append(store[n]['inp'][ok])
                acc_out[n].append(store[n]['out'][ok])

    for h in hooks:
        h.remove()

    imgs  = torch.cat(all_imgs).numpy()
    tgts  = torch.cat(all_tgts).numpy()
    confs = torch.cat(all_confs).numpy()

    layer_data = []
    for n, mod in named:
        is_conv = isinstance(mod, nn.Conv2d)
        layer_data.append({
            'name':        n,
            'type':        'conv' if is_conv else 'fc',
            'weight':      mod.weight.detach().cpu().numpy(),
            'input_fmap':  torch.cat(acc_inp[n]).numpy(),
            'output_fmap': torch.cat(acc_out[n]).numpy(),
        })
    return {'images': imgs, 'targets': tgts, 'confidences': confs, 'layer_data': layer_data}


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


def get_cifar10_loaders(batch_size=128, root='./data/', augment='baseline'):
    """Return (train_loader, test_loader) for CIFAR-10.

    augment: 'baseline' — RandomCrop + HorizontalFlip
             'cutout'   — baseline + CutOut(8×8 patch)
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
    elif augment == 'none':
        train_tfm = T.Compose(base)
    else:
        raise ValueError(f"augment must be 'baseline', 'cutout', or 'none'; got {augment!r}")

    test_tfm = T.Compose(base)

    train_ds = tv_datasets.CIFAR10(root, train=True,  download=True, transform=train_tfm)
    test_ds  = tv_datasets.CIFAR10(root, train=False, download=True, transform=test_tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=256,         shuffle=False)
    return train_loader, test_loader
