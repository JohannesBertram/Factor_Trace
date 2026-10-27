# Figures

Every new slide and paper figure, one section each. Write what needs to change under **Comments** of that figure; Claude works through them, applies the change, and clears the comment once it is done. If applicable Claude will use already present code for plotting components, such as `src/paper_figures.py` or `figures/fig1_overview/bftlib.py`, to seed the figure code.

Regenerate from the repo root with `PYTHONPATH=. .venv/bin/python figures/<script>.py`. Slides land in `figures/slides/` (dark), paper figures in `figures/paper/` (light), each with a PNG in `preview/`; the images below are those previews (`slides/preview/`, `paper/preview/`). A slide with builds writes `<name>_1`, `<name>_2`, ... on one canvas, so the builds overlay.

To ask for a new figure, add a section with its message as the heading and what it should show under **Comments**.

## Conventions

- Slides are dark (black background), paper figures light at AAAI text width. Every color comes from `style.COLOR`, whose keys are the components of `.figstyle/colors.yaml` (even / odd blue / red, BFT purple, ...).
- Slides carry no title inside the figure; the message is the section heading here.
- Little text, no overlapping text. One grey note line under a slide at most.
- Content is centered horizontally on every slide. Consecutive slides that show the same plot keep it at the same position; a slide that grows over builds centers its last build and aligns the earlier ones to it (`publish` in `style.py` centers each such group as one).
- Figures read only the `figures/figdata/` bundles (`figdata.load`, numpy + matplotlib only) or seeded toys.

## Slides

The method overview, in talk order, then the results. The method slides are all from `method.py`: a toy network with the sizes of the paper's even/odd MLP (8 x 4 x 2 on digits 0, 4, 1, 3), seeded activations and weights, real NMF fits on the toy arbors. Digit rows are ordered 0, 4, 1, 3; blue is even, red is odd, lighter shades are 4 and 3.

### activations: interpretability usually reads a layer's activations and maps them to concepts

![](slides/preview/activations.png)

After the inspiration deck: inputs into a network, one layer's activations read out as a vector and mapped to concepts.

**Comments:**

### arbor: the same activation can come from different inputs

![](slides/preview/arbor.png)

One neuron $n$ reached by a "0" and by a "1" through different weight x activation terms (edge width, one shade per input), stacked into the same sum (bars, facing each other across the =).

**Comments:**

### toy_task: an MLP sorts the digits 0, 1, 3, 4 into even and odd

![](slides/preview/toy_task.png)

**Comments:**

### overview: BFT asks what drives an output, splits it into circuits, and traces each circuit back

![](slides/preview/overview_1.png)
![](slides/preview/overview_2.png)

The toy_task slide, unchanged in layout, with the method drawn on it (the three slides overlay). Build 1: what drives each output splits into circuits, even (blue) and odd (red). Build 2: the even circuit traced back (green) splits again into a "0" and a "4" sub-circuit; the 0 and 4 inputs light up.

**Comments:**

### arbor_matrix: the arbor keeps every weight x activation term, one row per stimulus

![](slides/preview/arbor_matrix_1.png)
![](slides/preview/arbor_matrix_2.png)
![](slides/preview/arbor_matrix_3.png)

Builds: one stimulus into the even neuron, all stimuli, then the odd neuron's arbor appended as further columns.

**Comments:**

### nmf: NMF splits the arbor matrix into stimulus factors times connection factors

![](slides/preview/nmf.png)

**Comments:**

### simplification: for nonnegative inputs the arbor holds scaled copies of the activations, so a smaller matrix gives the same factors

![](slides/preview/simplification.png)

The rebuttal proposition (AI review W1), on the nmf slide's arbor matrix: each column $(i, j)$ of $J^+$ is activation column $j$ scaled by $W^+_{ij}$ (arrows; dashed where the weight is zero), so the stimulus x neuron matrix $\tilde A = A\,\mathrm{diag}(c)$ replaces it. Two separate NMF fits, one per matrix, give the same $W_\mathrm{stim}$, and the connection factor is read back per weight.

**Comments:**

### factor: each factor is a set of stimuli and the connections they use

![](slides/preview/factor_1.png)
![](slides/preview/factor_2.png)

Builds: the even factor, then the odd one. Edge width is the connection factor, node fill its mass per hidden neuron.

**Comments:**

### trace: a factor reweights the stimuli, and the layer below is factorized on them

![](slides/preview/trace_1.png)
![](slides/preview/trace_2.png)
![](slides/preview/trace_3.png)
![](slides/preview/trace_4.png)

Build 1: the network only; the even factor traces back into the layer below (green). Build 2: $\pi$ comes from the even factor's stimulus loadings: each bar is a stimulus's loadings on the even (blue) and odd (red) factor, summed to one, and $\pi$ is the even share (green box); 1 and 3 fall below the $\tau$ gate. Build 3: $\pi$ scales the rows of the arbor matrix one layer down. Build 4: NMF of that matrix gives a "0" and a "4" factor, drawn on the hidden-1 connections.

**Comments:**

### tree: repeating the step builds a factor tree

![](slides/preview/tree.png)

The tree as drawn on the fingerprint slides, so it stays put when they follow.

**Comments:**

### fingerprint: a stimulus's loadings across the tree are its fingerprint

![](slides/preview/fingerprint_1.png)
![](slides/preview/fingerprint_2.png)

Builds: the tree lit by one stimulus's loadings, then the fingerprints of all four stimuli. Toy loadings.

**Comments:**

### Results: the plan

From the paper's figures (main: Figs. 2, 4, 6, 8; appendix: A, B, E, G, N, P, fp_ood, fp_structure). All from `results.py`, drawn from the `figures/figdata/` bundles. The story, one message per slide:

1. **Circuits, small to large**: the even/odd MLP (Fig. 2a, c, d), the CIFAR-10 CNN (Fig. 6a, b), the pretrained ImageNet SqueezeNet (Fig. 8a, b). Same grammar each time: output factors are classes, one layer back they split into sub-groups.
2. **What changes with depth**: class purity falls to chance toward the input (Fig. 6d, 8d).
3. **The circuits are causal**: pruning the weights BFT ranks most important for a class, on all three models (Figs. 2e, 6f, 8c).
4. **Fingerprints**: class-structured (Fig. 4c) and more class-separable than the network's own activations (Fig. 4d, e, i).
5. **The weight term matters**: arbor NMF beats activation-only NMF on 5 of 5 models (Fig. P e); this is also the answer to "is it just activations?" after the simplification slide.
6. **Out of distribution**: held-out digits follow the network's own P(odd), and synthetic inputs collapse onto one fingerprint (Fig. fp_ood a, c).

The remaining results are backup slides at the end: the 10-digit MLP (Fig. B), the TinyViT (Fig. G), the full CNN and ImageNet galleries (Figs. E, N), conv1 color factors (Fig. 6c, e), faithfulness and stability (Fig. P a-c), rank sensitivity (Fig. P f), fingerprint baselines (Fig. P d), fingerprint class geometry (Fig. fp_structure a-e) and the OOD conditions (Fig. fp_ood f).

**Comments:**

### mlp_circuits: the even/odd MLP splits into two near-disjoint class circuits that refine into digit detectors

![](slides/preview/mlp_circuits_1.png)
![](slides/preview/mlp_circuits_2.png)

Fig. 2a, c, d. Build 1: both circuits in one graph (units grouped by circuit, node size = loading). Build 2: the $L_1$ factors of each circuit as pixel arbors, tagged with the digit each loads most.

**Comments:**

### mlp_inhibition: each circuit inhibits exactly where the other excites

![](slides/preview/mlp_inhibition.png)

Fig. 2a split by sign, the same graph twice (left aligned with mlp_circuits): excitatory edges stay inside a circuit, inhibitory edges (colored by the circuit they come from) cross to the other one, the push-pull at the output.

**Comments:**

### cnn_circuits: on a CIFAR-10 CNN every output factor is a class, and one layer back it splits into appearance groups

![](slides/preview/cnn_circuits_1.png)
![](slides/preview/cnn_circuits_2.png)

Fig. 6a, b. Build 1: the 13 output factors, top stimuli and top class. Build 2: car, horse and airplane traced one layer back (silver vs. red cars, horse poses, jets vs. airliners).

**Comments:**

### imagenet_circuits: the same holds unchanged on a pretrained 1000-way SqueezeNet

![](slides/preview/imagenet_circuits_1.png)
![](slides/preview/imagenet_circuits_2.png)

Fig. 8a, b. The 10 strongest output factors over 8 held-out categories; dog and car traced one fire module back.

**Comments:**

### depth: class identity dissolves toward the input

![](slides/preview/depth.png)

Fig. 6d, 8d. Class purity of every factor (dots) and the informativity-weighted mean (line), per layer.

**Comments:**

### pruning: pruning the traced weights collapses the class

![](slides/preview/pruning.png)

Figs. 2e, 6f, 8c. On ImageNet the most-important pruning is total rather than class-specific (both pink lines at zero), and least-important pruning is bimodal, so it is drawn per category.

**Comments:**

### fingerprint_classes: on CIFAR-10 each class has its own fingerprint block

![](slides/preview/fingerprint_classes.png)

Fig. 4c. Mean fingerprint per class, factors grouped by output circuit, each column scaled to its top class; the three circuits beyond one per class (a second ship, deer and airplane) sit at the right. All 143 columns: the paper's Fig. 4c drops those three blocks (`_cnn_fp_blocks` keeps one circuit per class) while its label says 143.

**Comments:**

### fingerprint_separability: the fingerprint separates classes better than the network's own activations

![](slides/preview/fingerprint_separability.png)

Fig. 4d, e, i. PCA of the CIFAR-10 fingerprint and penultimate activations; cosine silhouette for CIFAR-10 and ImageNet.

**Comments:**

### weight_term: factorizing weight x activation beats factorizing activations alone

![](slides/preview/weight_term.png)

Fig. P e. The same NMF and fingerprint, on the arbor vs. on the activations, all five models.

**Comments:**

### ood: the fingerprint tracks what the network believes, and flags inputs it does not recognize

![](slides/preview/ood.png)

Fig. fp_ood c, a. Left: held-out digits land where the network's own P(odd) puts them, not at their true parity. Right: synthetic inputs (noise, gray, checkerboard, inverted) collapse onto one fingerprint on every model; ImageNet has no held-out set.

**Comments:**

### Backup slides

All from `results.py`, after the main results.

#### backup_digit_mlp: on ten digits, output factors pool several digits and tracing back un-pools them

![](slides/preview/backup_digit_mlp.png)

Fig. B a, b, g. Output factor x digit share; best digit purity per circuit from $L_3$ to $L_1$; the $L_1$ detector of each digit pooled by $f_0$ and $f_1$.

**Comments:**

#### backup_vit: a TinyViT block traces like an MLP, its circuits pooling similar digits along parity

![](slides/preview/backup_vit.png)

Fig. G b, f. Output circuit x digit (label color: the parity it leans to) and the strongest stimuli of each circuit.

**Comments:**

#### backup_cnn_gallery: every CIFAR-10 class circuit and its two strongest sub-factors

![](slides/preview/backup_cnn_gallery.png)

Fig. E. The strongest output factor of each class, its top stimuli, and the top stimuli of its two strongest sub-factors one layer back.

**Comments:**

#### backup_imagenet_gallery: the same on ImageNet

![](slides/preview/backup_imagenet_gallery.png)

Fig. N. One circuit per category (8 categories).

**Comments:**

#### backup_conv1: at conv1 the factors are color channels

![](slides/preview/backup_conv1.png)

Fig. 6c, e. The conv1 factors under the cat circuit with their input mass per RGB channel; the color spread of each layer's top stimuli falls toward the input.

**Comments:**

#### backup_validation: the decomposition is faithful and stable

![](slides/preview/backup_validation.png)

Fig. P a-c. Causal reconstruction $R^2$, NNLS round-trip and NMF-seed stability per model; n/a where a model cannot supply the measure.

**Comments:**

#### backup_rank: the factors barely depend on the exact rank

![](slides/preview/backup_rank.png)

Fig. P f. Cosine of each node's factors at $K^* \pm 1$ to those at $K^*$ (dots: nodes, bar: median).

**Comments:**

#### backup_fp_baselines: the fingerprint against activation baselines at matched width

![](slides/preview/backup_fp_baselines.png)

Fig. P d. On the even/odd MLP the matched-width activations beat the fingerprint (3 of 5 models in favor of the fingerprint, as in the paper).

**Comments:**

#### backup_fp_geometry: on every model the fingerprint geometry is block-diagonal by class

![](slides/preview/backup_fp_geometry.png)

Fig. fp_structure a-e. Pairwise cosine of 30 test fingerprints per class (digit for the parity models), sorted by class; silhouette on all test fingerprints.

**Comments:**

#### backup_ood_conditions: synthetic inputs load a few factors, real images spread over all circuits

![](slides/preview/backup_ood_conditions.png)

Fig. fp_ood f. CIFAR-10 mean fingerprint per condition.

**Comments:**

## Paper
