"""BFT dataclasses — standardised output types for the Backward Factor Trace pipeline."""

from dataclasses import dataclass, field
from collections import deque
from typing import Optional

import numpy as np


@dataclass
class BFTNode:
    """Single-layer NMF factorization node in the BFT tree.

    The tree is rooted at the output layer; children point toward the input.
    Leaves (children=[]) correspond to the input-side layer.

    Fields
    ------
    layer_idx         : int — 0 = input-side layer (leaves), max = output layer (root)
    layer_name        : str — module name from the model
    layer_type        : 'fc' | 'conv' | 'attn'
    path              : tuple — branch path from root, e.g. (0, 1) means first
                        branch at root, second branch from there
    weight            : (n_out, n_in) or (C_out, C_in, kH, kW) — layer weight matrix
    img_factors       : (N, K) — per-stimulus NMF loadings
    connection_factors: (n_features, K) — arbor-space NMF components (H matrix);
                        n_features = n_out * n_in for FC, C_out*C_in*kH*kW for conv
    lambdas           : (K,) — component magnitudes, sorted descending
    stimulus_weights  : (N,) — per-stimulus relevance weights inherited from parent
    neg_img_factors   : (N, K) or None — loadings for the inhibitory factorisation
    neg_connection_factors : (n_features, K) or None — inhibitory NMF components
    neg_lambdas       : (K,) or None — inhibitory component magnitudes
    weighting         : str — weighting strategy used during tracing
    stimulus_threshold: float — stimulus quantile threshold applied during tracing
    attn_weights      : (N, T) or None — CLS-row attention scores (attn layers only)
    children          : list[BFTNode] — child nodes toward the input; empty at leaves
    """
    layer_idx: int
    layer_name: str
    layer_type: str
    path: tuple
    weight: np.ndarray
    img_factors: np.ndarray
    connection_factors: np.ndarray
    lambdas: np.ndarray
    stimulus_weights: np.ndarray
    neg_img_factors: Optional[np.ndarray] = None
    neg_connection_factors: Optional[np.ndarray] = None
    neg_lambdas: Optional[np.ndarray] = None
    weighting: str = 'img_selectivity'
    stimulus_threshold: float = 0.0
    attn_weights: Optional[np.ndarray] = None
    children: list = field(default_factory=list)

    @property
    def factor_idx(self) -> int:
        """Which factor of the parent node spawned this branch (path[-1])."""
        return int(self.path[-1]) if self.path else 0


@dataclass
class BFTResult:
    """Full output of bft(). Contains the factor tree and stimulus metadata.

    Fields
    ------
    root        : BFTNode — root of the BFT tree (output-side layer)
    images      : (N, ...) — input stimuli collected during the trace
    targets     : (N,) — class labels
    confidences : (N,) — model confidence on the predicted class
    """
    root: BFTNode
    images: np.ndarray
    targets: np.ndarray
    confidences: np.ndarray

    def nodes(self) -> list:
        """Return all BFTNodes in BFS order (root first, leaves last)."""
        out, q = [], deque([self.root])
        while q:
            n = q.popleft()
            out.append(n)
            q.extend(n.children)
        return out

    @property
    def n_samples(self) -> int:
        return len(self.targets)


@dataclass
class FingerprintResult:
    """Per-stimulus representation derived from a BFT trace.

    Fields
    ------
    matrix     : (N, D) — concatenated img_factors across all BFT nodes (BFS order);
                 D = sum of K_i over all nodes
    similarity : (N, N) — pairwise cosine similarity matrix
    indices    : (N,) — which sample indices from the source BFTResult were used
    """
    matrix: np.ndarray
    similarity: np.ndarray
    indices: np.ndarray
