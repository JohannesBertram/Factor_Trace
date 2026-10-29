# Figures

Every new slide and paper figure, one section each. Write what needs to change under **Comments** of that figure; Claude works through them, applies the change, and clears the comment once it is done. If applicable Claude will use already present code for plotting components, such as `src/paper_figures.py` or `figures/fig1_overview/bftlib.py`, to seed the figure code.

Regenerate from the repo root with `PYTHONPATH=. .venv/bin/python figures/<script>.py`. Slides land in `figures/slides/` (dark), paper figures in `figures/paper/` (light), each with a PNG in `preview/`; the images below are those previews (`slides/preview/`, `paper/preview/`). A slide with builds writes `<name>_1`, `<name>_2`, ... on one canvas, so the builds overlay.

To ask for a new figure, add a section with its message as the heading and what it should show under **Comments**.

Each slide section has a **Notes:** block: the presenter notes that the sync-slides skill writes into the Keynote deck. A paragraph starting with `Build k:` goes to the k-th build of the section only; every other paragraph goes to all its builds. Notes edited in Keynote are never overwritten; clear them there to get the ones from here back.

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

**Notes:**

Most interpretability work reads a layer's activations: we push images or text through the network, take one layer's activation vector, and map it to concepts like a dog, the Golden Gate Bridge or unsafe behavior. That treats the activation as the unit of meaning. But an activation is only the result of a computation, the sum of incoming weights times the activations of the layer before. Reading activations alone throws away which weights produced them.

**Comments:**

### arbor: the same activation can come from different inputs

![](slides/preview/arbor.png)

One neuron $n$ reached by a "0" and by a "1" through different weight x activation terms (edge width, one shade per input), stacked into the same sum (bars, facing each other across the =).

**Notes:**

Here is why that matters. Neuron n responds to a 0 and to a 1 with the same activation: the stacked bars on both sides of the equals sign have the same height. But the 0 drives it mostly through the left input and the 1 through the middle one. The individual weight x activation terms differ, only their sum is the same. From the activation alone the two cases look identical; BFT keeps the individual terms, so it can tell them apart.

**Comments:**

### toy_task: an MLP sorts the digits 0, 1, 3, 4 into even and odd

![](slides/preview/toy_task.png)

**Notes:**

To explain the method I use a toy network with the sizes of the even/odd MLP from the paper: 28 x 28 pixel inputs, hidden layers of 8 and 4 units, and two outputs, even and odd. It is trained on the digits 0, 1, 3 and 4. On the method slides blue is even and red is odd; the lighter shades mark the 4 and the 3.

**Comments:**

### overview: BFT asks what drives an output, splits it into circuits, and traces each circuit back

![](slides/preview/overview_1.png)
![](slides/preview/overview_2.png)

The toy_task slide, unchanged in layout, with the method drawn on it (the three slides overlay). Build 1: what drives each output splits into circuits, even (blue) and odd (red). Build 2: the even circuit traced back (green) splits again into a "0" and a "4" sub-circuit; the 0 and 4 inputs light up.

**Notes:**

Build 1: BFT starts at the output. It asks what drives each output unit across all stimuli and splits that into a few factors. Here it finds two: an even circuit in blue, carried by two of the four hidden units, and an odd circuit in red, carried by the other two. No labels are used; the split comes from the weights and activations alone.

Build 2: Then each factor is traced backward. Followed one layer down (green arrows), the even circuit splits again, into a part used by the 0s and a part used by the 4s, so the 0 and the 4 light up on the input side. Repeating this layer by layer gives a tree of circuits from the output down to the input, and that tree is what BFT returns.

**Comments:**

### arbor_matrix: the arbor keeps every weight x activation term, one row per stimulus

![](slides/preview/arbor_matrix_1.png)
![](slides/preview/arbor_matrix_2.png)
![](slides/preview/arbor_matrix_3.png)

Builds: one stimulus into the even neuron, all stimuli, then the odd neuron's arbor appended as further columns.

**Notes:**

Build 1: Now the method step by step. Take one neuron, here the even output, and one stimulus, a 0. Instead of summing its inputs, we keep each incoming term, weight times activation, as a separate entry. That gives one row of numbers: the first row of the arbor matrix J, under the even neuron's block of columns.

Build 2: We do the same for every stimulus. Each stimulus is a row and each column is one incoming connection of the even neuron. The 0s and 4s drive it strongly through the first connections; the 1s and 3s barely drive it at all.

Build 3: And we do it for every neuron in the layer, appending the odd neuron's connections as further columns. The result is the arbor matrix of the layer: stimuli by connections, every entry a weight x activation term. It records not only which neurons respond, but through which connections.

**Comments:**

### nmf: NMF splits the arbor matrix into stimulus factors times connection factors

![](slides/preview/nmf.png)

**Notes:**

We factorize the arbor matrix with non-negative matrix factorization. NMF writes it as a sum of a few rank-one parts, each a stimulus factor (which stimuli use this part) times a connection factor (which connections they use). Because everything is non-negative, the parts only add up and never cancel, which keeps each one readable. Here two factors suffice: f0 holds the 0s and 4s with the even connections, f1 the 1s and 3s with the odd ones. The negative terms, inhibition, are factorized separately in the same way.

**Comments:**

### simplification: for nonnegative inputs the arbor holds scaled copies of the activations, so a smaller matrix gives the same factors

![](slides/preview/simplification.png)

The rebuttal proposition (AI review W1), on the nmf slide's arbor matrix: each column $(i, j)$ of $J^+$ is activation column $j$ scaled by $W^+_{ij}$ (arrows; dashed where the weight is zero), so the stimulus x neuron matrix $\tilde A = A\,\mathrm{diag}(c)$ replaces it. Two separate NMF fits, one per matrix, give the same $W_\mathrm{stim}$, and the connection factor is read back per weight.

**Notes:**

In practice the arbor matrix is huge, with one column per weight. But for non-negative inputs, as after a ReLU, each column is just an activation column scaled by its weight (the arrows, dashed where the weight is zero). So we can factorize a much smaller matrix, stimuli by input neurons, with each activation column scaled by the norm of that neuron's outgoing weights. It gives the same stimulus factors, and the connection factor is read back per weight afterwards. Note that this is still not just activations: the weight scaling changes what NMF finds, and we test that directly in the results.

**Comments:**

### factor: each factor is a set of stimuli and the connections they use

![](slides/preview/factor_1.png)
![](slides/preview/factor_2.png)

Builds: the even factor, then the odd one. Edge width is the connection factor, node fill its mass per hidden neuron.

**Notes:**

Build 1: This is what a factor looks like. On the left the stimulus factor: the 0 and the 4 load on f0, the 1 and the 3 do not. On the right the connection factor drawn on the network: edge width is the factor's weight on each connection, node fill its mass per hidden neuron. So f0 is an even circuit: the 0s and 4s, plus the two hidden units and connections through which they reach the even output.

Build 2: And the second factor, f1, is the odd circuit: the 1s and 3s, through the other two hidden units into the odd output. One factorization of one layer already separates the network's two decisions into two sets of stimuli, each with the connections it uses.

**Comments:**

### trace: a factor reweights the stimuli, and the layer below is factorized on them

![](slides/preview/trace_1.png)
![](slides/preview/trace_2.png)
![](slides/preview/trace_3.png)
![](slides/preview/trace_4.png)

Build 1: the network only; the even factor traces back into the layer below (green). Build 2: $\pi$ comes from the even factor's stimulus loadings: each bar is a stimulus's loadings on the even (blue) and odd (red) factor, summed to one, and $\pi$ is the even share (green box); 1 and 3 fall below the $\tau$ gate. Build 3: $\pi$ scales the rows of the arbor matrix one layer down. Build 4: NMF of that matrix gives a "0" and a "4" factor, drawn on the hidden-1 connections.

**Notes:**

Build 1: To get a full circuit through the network, we trace each output factor backward. Take the even factor: we want to know how the layer below computes what feeds it. So we go one layer down (green) and factorize that layer's arbor, but only for the stimuli the even factor cares about.

Build 2: Which stimuli count is set by a weight pi. Each bar is one stimulus's loadings on the even (blue) and the odd (red) factor, normalized to sum to one; pi is the even share, boxed in green. The 0s and 4s get almost all of their weight from the even factor. The 1s and 3s fall below the tau gate and are dropped.

Build 3: pi then scales the rows of the arbor matrix one layer down. The stimuli that belong to the even factor dominate and the others are gone, so the next factorization only sees what the even circuit computes.

Build 4: NMF of that reweighted matrix splits the even circuit into two sub-circuits, a 0 factor and a 4 factor, each with its own connections in the first hidden layer. That is one step of the backward trace.

**Comments:**

### tree: repeating the step builds a factor tree

![](slides/preview/tree.png)

The tree as drawn on the fingerprint slides, so it stays put when they follow.

**Notes:**

Repeating that step at every layer, for every factor, builds a tree. The root is the output layer, here split into even and odd. Each child is the factorization of the layer below under its parent's stimulus weights: even splits into 0 and 4, odd into 1 and 3, and so on down to the input. Each path through the tree is a circuit, a set of stimuli plus the connections they use at every layer. The number of factors per layer is chosen by held-out reconstruction, not by hand.

**Comments:**

### fingerprint: a stimulus's loadings across the tree are its fingerprint

![](slides/preview/fingerprint_1.png)
![](slides/preview/fingerprint_2.png)

Builds: the tree lit by one stimulus's loadings, then the fingerprints of all four stimuli. Toy loadings.

**Notes:**

Build 1: The tree also gives a code for each stimulus. Take a single 0: its loadings on the nodes of the tree's top two levels, even and odd and their sub-circuits, form a vector. That vector is the stimulus's fingerprint: it describes how this stimulus is routed through the network.

Build 2: Stacking the fingerprints of all four stimuli gives this matrix: the even digits load on the even side and their own sub-circuit, the odd digits on the odd side. A new stimulus can be projected onto the fixed factors without refitting, which we use later for held-out and out-of-distribution inputs. These are toy loadings; the real ones follow.

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

**Notes:**

Build 1: Now the real even/odd MLP. Without labels, BFT splits it into two circuits, even in blue and odd in red, that share almost no units: each hidden unit belongs to one circuit. Node size is the loading; solid lines are excitatory, dotted ones inhibitory.

Build 2: Traced down to the pixels, each circuit refines into digit detectors. These are the first-layer factors of each circuit as pixel arbors, tagged with the digit that loads most: the even circuit's factors look like 4s and 0s, the odd circuit's like 1s and 3s. So the network computes parity by first detecting the individual digits and then pooling them.

**Comments:**

### mlp_inhibition: each circuit inhibits exactly where the other excites

![](slides/preview/mlp_inhibition.png)

Fig. 2a split by sign, the same graph twice (left aligned with mlp_circuits): excitatory edges stay inside a circuit, inhibitory edges (colored by the circuit they come from) cross to the other one, the push-pull at the output.

**Notes:**

The same graph, split by sign. Left, the excitatory connections: they stay inside a circuit. Right, the inhibitory ones, colored by the circuit they come from: they cross over into the other circuit. Each circuit pushes its own output up and the other output down, a push-pull at the output.

**Comments:**

### cnn_circuits: on a CIFAR-10 CNN every output factor is a class, and one layer back it splits into appearance groups

![](slides/preview/cnn_circuits_1.png)
![](slides/preview/cnn_circuits_2.png)

Fig. 6a, b. Build 1: the 13 output factors, top stimuli and top class. Build 2: car, horse and airplane traced one layer back (silver vs. red cars, horse poses, jets vs. airliners).

**Notes:**

Build 1: Next, a CNN trained on CIFAR-10. The output layer factorizes into 13 factors, shown with their top stimuli and the class that dominates them. Every factor is a class, again found without labels; three classes (ship, deer and plane) get a second factor.

Build 2: One layer back, each class circuit splits into appearance groups: the car circuit into silver and red cars, the horse circuit by pose, the plane circuit into jets and airliners. A class is assembled from several visually distinct sub-circuits.

**Comments:**

### imagenet_circuits: the same holds unchanged on a pretrained 1000-way SqueezeNet

![](slides/preview/imagenet_circuits_1.png)
![](slides/preview/imagenet_circuits_2.png)

Fig. 8a, b. The 10 strongest output factors over 8 held-out categories; dog and car traced one fire module back.

**Notes:**

Build 1: The same method, unchanged, on a pretrained 1000-way SqueezeNet, evaluated on 8 coarse categories that each group several ImageNet classes. These are the 10 strongest output factors: each lines up with one category (dog, plane, car, bicycle, bird, elephant, ship, bear), even though these categories were never training targets.

Build 2: Traced back one fire module, the split again stays within the category and never crosses it. The dog circuit splits by scene, close-up dogs versus dogs in grass, and the car circuit into the category's two ImageNet classes, jeep and race car.

**Comments:**

### depth: class identity dissolves toward the input

![](slides/preview/depth.png)

Fig. 6d, 8d. Class purity of every factor (dots) and the informativity-weighted mean (line), per layer.

**Notes:**

What changes as we trace toward the input? Class purity: how much of a factor's top stimuli come from one class. Each dot is a factor and the line is the informativity-weighted mean per layer. On CIFAR-10 purity is about 0.5 near the output and falls to 0.17 to 0.19 in the early layers, against a chance level of 0.1; on ImageNet it falls from 0.58 to about 0.2. Near the input the factors are shared features, not class detectors.

**Comments:**

### pruning: pruning the traced weights collapses the class

![](slides/preview/pruning.png)

Figs. 2e, 6f, 8c. On ImageNet the most-important pruning is total rather than class-specific (both pink lines at zero), and least-important pruning is bimodal, so it is drawn per category.

**Notes:**

Are these circuits the ones the network actually uses? We prune a class circuit's weights in order of BFT importance and measure accuracy. Solid line: the circuit's own class, which collapses after only a few percent of the weights. Dashed: the other classes, which drop much less on the MLP and the CNN. Dotted: pruning the least important or random weights, which hurts far less. On ImageNet the ranking is causal but not class-specific: pruning one category's most important weights takes every category down at 2%. The least-important pruning is bimodal there, so it is drawn per category.

**Comments:**

### fingerprint_classes: on CIFAR-10 each class has its own fingerprint block

![](slides/preview/fingerprint_classes.png)

Fig. 4c. Mean fingerprint per class, factors grouped by output circuit, each column scaled to its top class; the three circuits beyond one per class (a second ship, deer and airplane) sit at the right. All 143 columns: the paper's Fig. 4c drops those three blocks (`_cnn_fp_blocks` keeps one circuit per class) while its label says 143.

**Notes:**

Back to the fingerprint, on the CIFAR-10 CNN. Each row is the mean fingerprint of one class; the columns are the 143 factors of the tree's top two levels, grouped by output circuit, each scaled to its top class. The matrix is block-diagonal: each class loads on its own circuit and that circuit's sub-factors. The three blocks at the right are the second circuits of ship, deer and airplane.

**Comments:**

### fingerprint_separability: the fingerprint separates classes better than the network's own activations

![](slides/preview/fingerprint_separability.png)

Fig. 4d, e, i. PCA of the CIFAR-10 fingerprint and penultimate activations; cosine silhouette for CIFAR-10 and ImageNet.

**Notes:**

How good is this as a class code compared with what one normally uses, the network's penultimate activations? Left, a PCA of both on the CIFAR-10 test set, colored by class: the fingerprint clusters are tighter. Right, the cosine silhouette: 0.67 versus 0.36 on CIFAR-10 and 0.43 versus 0.21 on ImageNet. On both convolutional models the fingerprint is the more class-separable code.

**Comments:**

### weight_term: factorizing weight x activation beats factorizing activations alone

![](slides/preview/weight_term.png)

Fig. P e. The same NMF and fingerprint, on the arbor vs. on the activations, all five models.

**Notes:**

Is this just NMF on activations? We ran the same pipeline, node for node and at the same ranks, on the activations alone, without the weight term. On all five models the arbor version is more class-separable (one-sided Wilcoxon p = 0.031). The gap is largest on the MLPs and smaller, but never reversed, on the deeper networks. So the weights carry information that the activations alone do not.

**Comments:**

### ood: the fingerprint tracks what the network believes, and flags inputs it does not recognize

![](slides/preview/ood.png)

Fig. fp_ood c, a. Left: held-out digits land where the network's own P(odd) puts them, not at their true parity. Right: synthetic inputs (noise, gray, checkerboard, inverted) collapse onto one fingerprint on every model; ImageNet has no held-out set.

**Notes:**

Finally, out-of-distribution inputs. Left, the even/odd MLP on digits it never saw: their share on the odd circuit tracks the network's own probability of saying odd (r = 0.999), not their true parity. The fingerprint reflects what the network believes. Right, synthetic inputs (noise, gray images, checkerboards, inverted images) collapse onto almost the same fingerprint on every model, with cosine to their condition mean above 0.9, while real images stay spread out. That collapse flags inputs the network does not recognize. ImageNet has no held-out set, so that bar is missing.

**Comments:**

### Backup slides

All from `results.py`, after the main results.

#### backup_digit_mlp: on ten digits, output factors pool several digits and tracing back un-pools them

![](slides/preview/backup_digit_mlp.png)

Fig. B a, b, g. Output factor x digit share; best digit purity per circuit from $L_3$ to $L_1$; the $L_1$ detector of each digit pooled by $f_0$ and $f_1$.

**Notes:**

Backup: a 784-40-20-10 MLP on all ten digits. Left, the output factors do not map to single digits; each pools several, and the largest digit share is only 0.28 to 0.54. Middle, tracing a circuit back un-pools them: digit purity rises toward the input. Right, an example: f0 pools 9, 8 and 0, and its first-layer factors are separate detectors for each of those digits.

**Comments:**

#### backup_vit: a TinyViT block traces like an MLP, its circuits pooling similar digits along parity

![](slides/preview/backup_vit.png)

Fig. G b, f. Output circuit x digit (label color: the parity it leans to) and the strongest stimuli of each circuit.

**Notes:**

Backup: a TinyViT trained on even versus odd. BFT factorizes the block's projections like MLP layers. Each output circuit pools a few similar-looking digits; the digit labels are colored by the parity they lean to. Right, the strongest stimuli of each circuit, each dominated by one or two digit shapes.

**Comments:**

#### backup_cnn_gallery: every CIFAR-10 class circuit and its two strongest sub-factors

![](slides/preview/backup_cnn_gallery.png)

Fig. E. The strongest output factor of each class, its top stimuli, and the top stimuli of its two strongest sub-factors one layer back.

**Notes:**

Backup: the full CIFAR-10 gallery. For each class, the strongest output factor with its top stimuli, and below it the top stimuli of its two strongest sub-factors one layer back. Every class splits into appearance groups, like the cars and horses on the main slides.

**Comments:**

#### backup_imagenet_gallery: the same on ImageNet

![](slides/preview/backup_imagenet_gallery.png)

Fig. N. One circuit per category (8 categories).

**Notes:**

Backup: the same gallery for the SqueezeNet, one circuit for each of the eight categories, with its two strongest sub-factors one fire module back. The sub-factors stay within their category and split it by scene, view or subclass, for example bear species and airplane views.

**Comments:**

#### backup_conv1: at conv1 the factors are color channels

![](slides/preview/backup_conv1.png)

Fig. 6c, e. The conv1 factors under the cat circuit with their input mass per RGB channel; the color spread of each layer's top stimuli falls toward the input.

**Notes:**

Backup: what happens at the very first layer. Left, the conv1 factors under the cat circuit, with their input mass per RGB channel as the colored strip: they are largely color channels. Right, the color spread of each layer's top stimuli falls toward the input and stays below random, so the early factors group stimuli by color, not by class.

**Comments:**

#### backup_validation: the decomposition is faithful and stable

![](slides/preview/backup_validation.png)

Fig. P a-c. Causal reconstruction $R^2$, NNLS round-trip and NMF-seed stability per model; n/a where a model cannot supply the measure.

**Notes:**

Backup: is the decomposition faithful? Left, causal reconstruction R-squared, how much of each node's pre-activation the factors reconstruct: 0.96 and 0.72 on the two MLPs, n/a where a model cannot supply the measure. Middle, the NNLS round-trip: held-out stimuli re-projected onto the fixed factors keep a median cosine of 0.74 to 1.0. Right, stability across NMF seeds: 0.86 to 0.95, all above our 0.85 gate.

**Comments:**

#### backup_rank: the factors barely depend on the exact rank

![](slides/preview/backup_rank.png)

Fig. P f. Cosine of each node's factors at $K^* \pm 1$ to those at $K^*$ (dots: nodes, bar: median).

**Notes:**

Backup: sensitivity to the rank. For every node we refit with one factor fewer and one more than the chosen rank K* and compare the factors with those at K*. The cosine stays near 1 on all five models (dots are nodes, bars the median), with one outlier at K* - 1 on the even/odd MLP.

**Comments:**

#### backup_fp_baselines: the fingerprint against activation baselines at matched width

![](slides/preview/backup_fp_baselines.png)

Fig. P d. On the even/odd MLP the matched-width activations beat the fingerprint (3 of 5 models in favor of the fingerprint, as in the paper).

**Notes:**

Backup: the fingerprint against more baselines: the network's activations at native width, reduced by PCA to the fingerprint's width, and randomly projected to that width. The fingerprint is clearly better on 3 of 5 models (digit MLP, CIFAR-10 CNN and ImageNet). On the even/odd MLP the matched-width activations beat it, and on the TinyViT all codes are about equal.

**Comments:**

#### backup_fp_geometry: on every model the fingerprint geometry is block-diagonal by class

![](slides/preview/backup_fp_geometry.png)

Fig. fp_structure a-e. Pairwise cosine of 30 test fingerprints per class (digit for the parity models), sorted by class; silhouette on all test fingerprints.

**Notes:**

Backup: the fingerprint's class geometry on every model. Pairwise cosine between 30 test fingerprints per class, sorted by class (by digit for the two parity models): block-diagonal throughout. The silhouette on the full test set is under each matrix.

**Comments:**

#### backup_ood_conditions: synthetic inputs load a few factors, real images spread over all circuits

![](slides/preview/backup_ood_conditions.png)

Fig. fp_ood f. CIFAR-10 mean fingerprint per condition.

**Notes:**

Backup: where the OOD collapse comes from, on CIFAR-10. This is the mean fingerprint per condition over the 143 factors. Test images and CIFAR-100 images spread over all circuits; noise, gray, checkerboard and inverted images load on only a few factors, so they all end up at nearly the same fingerprint.

**Comments:**

## Paper
