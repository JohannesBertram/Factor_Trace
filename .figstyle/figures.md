# Figure log

One entry per figure: key message + final caption. Keep in sync with edits.

Venue: AAAI (tueplots `aaai2024` geometry: 6.975 in text width, 3.3 in column width, 9 pt body, Times via `\usepackage{times}`). Include with `\includegraphics{...}` and NO `width=` — figures are produced at final size.

## fig2_mlp_circuits.pdf  (main text, Fig. 2, full width)

source: `notebooks/01_MLP_8_4_0134.ipynb`, last code cell but one

message: BFT splits the parity net into two class circuits that barely share hidden units and push-pull at the output, and each circuit refines into digit-selective sub-circuits toward the pixels — the digit split just happens at different depths in the two circuits.

caption: \textbf{BFT splits the $784\to8\to4\to2$ even/odd MLP into two near-disjoint class circuits that refine into digit-selective sub-circuits toward the input.} \textbf{(a)} Both circuits in one graph (node area $\propto$ loading, unit order sorted by owning circuit; solid excitatory, dashed inhibitory): they claim different $L_1$ units (support cosine $0.27$ vs $0.59$ for a within-circuit shuffle, $p=0.015$; quantified in Fig.~A) and each drives its own logit while suppressing the other. \textbf{(b)} Loading over the four digits (ordered 0,\,4,\,1,\,3) of the $L_3$ and $L_2$ factors, bars in black; the blue/red titles name the circuit and the tree links each $L_3$ factor to its $L_2$ children (the even $L_3$ factor centered above its two). The even circuit already splits into a ``4'' and a ``0'' factor at $L_2$; the odd circuit does not split until $L_1$. \textbf{(c,~d)} Pixel arbor and digit loading of the four $L_1$ factors below each circuit's $L_2\,f_0$ (bold: the digit the factor loads most): 4- and 0-detectors on the even side, 1- and 3-detectors on the odd side. Digit purity rises monotonically toward the input across the trace ($0.49\to0.63\to0.82$ against $0.25$ chance; quantified in Fig.~A).

## (removed) fig3_digit_mlp_circuits — merged into figB_digit_mlp_details

The digit-MLP circuits result (no space for it as a main figure) is now the first half of the merged appendix figure below.

## appendix/figB_digit_mlp_details.pdf  (appendix, full width) — merged main + details

source: `notebooks/02_MLP_40_20_digits.ipynb` (bundle `nb02_circuits`)

message: The ten-class net does not factorize into digits at the output — it factorizes into seven distributed, heavily overlapping circuits that each pool several digits, and tracing a circuit backward un-pools it into digit-selective factors. Merges the former main figure with its decomposition detail.

caption: \textbf{Class circuits in the $784\to40\to20\to10$ digit MLP: the output layer factorizes into seven circuits that each pool several digits, and tracing a circuit to the pixels un-pools it.} \textbf{(a)} Per-digit loading of the seven output-layer factors --- the largest digit share is only $0.28$--$0.54$, so no output factor is a digit detector. \textbf{(b)} Digit purity along each circuit (grey: every factor; thin lines: each circuit's purest factor; thick: their mean): the purest rises $0.41\to0.75\to0.77$ while the spread stays wide. $20$ of $21$ pooled digits get an $L_1$ factor of their own, against $6$ under another circuit's factors. \textbf{(c)} The circuits are distributed, unlike Fig.~2: $25$ of $40$ $L_1$ units per circuit, pairwise support overlap $0.77$, \emph{above} the $0.63$ shuffle null. \textbf{(d)} $\lambda$ spectra of the output node and the $L_1$ node of every circuit. \textbf{(e)} Each circuit through the full network; of its $\approx1000$ edges only each unit's strongest input is drawn. \textbf{(f)} For each circuit, the pixel arbor of the $L_1$ factor that best detects its first, second and third pooled digit (bold: that digit; $f$: which of the circuit's factors) --- e.g.\ $f_3$ pools 2,\,0,\,7 and supplies a 2-, a 0- and a 7-detector.

## appendix/figA_mlp_details.pdf  (appendix, full width)

source: `notebooks/01_MLP_8_4_0134.ipynb`, last code cell

message: The Fig. 2 decomposition is well conditioned: a graded spectrum per node, an output layer wired as clean push-pull, hidden-unit supports that barely touch, sign-mirrored connection maps, and driving stimuli that confirm what each L1 factor detects.

caption: \textbf{Decomposition details for the $8\times4$ even/odd MLP.} \textbf{(a)} $\lambda$ spectrum of every node of the trace. \textbf{(b)} Each output-layer factor's arbor mass, split into what it excites and what it inhibits: the even factor drives the even logit and suppresses the odd one, and vice versa, with excitation and inhibition of comparable magnitude. \textbf{(c)} How the two circuits divide the eight $L_1$ units (support cosine $0.27$ against a shuffle null of $0.59\pm0.15$): units~1--4 are even-only, units~0 and~6 odd-only, and only units~5 and~7 carry both. \textbf{(d)} Connection maps (out $\times$ in) at $L_3$ and $L_2$: the inhibitory map mirrors the excitatory one. \textbf{(e,~f)} Weighted-average driving stimulus of every $L_1$ factor (dark = high); the bold digit is the one the factor loads most.

## fig4_fingerprints_main.pdf  (main text, full width) — the single fingerprint figure

source: merged render (reads `nb03_fingerprints`, and loads `nb01_fingerprints` for the intro panels a, b)

message: A BFT fingerprint is the vector of factor loadings a stimulus evokes. On the legible 8x4 even/odd MLP the trace's 13 factors form a tree (a) and the fingerprint is one entry per node of that tree, giving a class-structured code (b); on the CIFAR-10 CNN the 156-d fingerprint stays class-structured (c) and is as class-separable as the network's own 256-d penultimate activations (d, e), with a clean block class geometry (f); and on a pretrained 1000-way ImageNet SqueezeNet the 253-d fingerprint separates 8 held-out categories (g) where the network's own 512-d penultimate representation does not (h), beating the dimension-matched activations by 4x (i) — the single fingerprint main figure of the paper, absorbing the former fig9.

NOTE panel (h)'s activation baseline comes from notebook 05 §5, which writes the `act` entry itself — pooled from the layer dicts it already collected in §2, so it is row-aligned with the fingerprints by construction (`scripts/add_activation_baselines.py` cannot supply this one: it would need the ImageNet val images). Two reps are stored, `all layers act.` (2624-d, every squeeze-spine layer global-average-pooled and concatenated, silhouette 0.11) and `classifier act.` (512-d, silhouette 0.21); the figure uses `reps[-1]`, the classifier. On a bundle exported before 2026-07-24 the panel is an empty dashed frame instead.

The silhouettes in (d)/(e) and (g)/(h) are cosine silhouettes on each representation at its own native dimension. Panel (i) simply reprints those four numbers as bars (fp 0.54/0.51, act 0.44/0.21) — it is NOT the dimension-matched validation measurement anymore. The fair, dimension-matched comparison across settings (full population, last-layer only, PCA dim-matched, random projection) lives in the new appendix Fig.~P, so the main figure states the native-dimension result and the appendix carries the controls. Both readings agree in ordering.

Every panel label names the dataset it comes from (MNIST / CIFAR-10 / ImageNet, and "both" for the head-to-head in (i)), because the rows are not one model each — row 1 crosses from the MNIST MLP to the CIFAR-10 CNN.

caption: \textbf{The BFT fingerprint: what it is, and that it is a class-structured, class-separable code.} \textbf{(a)} Where the entries of a fingerprint come from, on the legible $784\to8\to4\to2$ MNIST MLP: the trace's 13 factors as the tree they form --- the output layer's two factors (blue even, red odd) branch into three $L_2$ factors (one of which stays a leaf) and eight $L_1$ factors. The same tree is drawn twice, once per stimulus digit, with node area and shade $\propto$ how strongly that digit drives the factor on one shared scale: a 0 lights the even branch and leaves the odd one pale, a 1 does the reverse. That pattern of node shades, read off in order, \emph{is} the fingerprint --- panel (b) is the same quantity as a heatmap. \textbf{(b)} The fingerprint itself: mean loading per digit over those same 13 factors, columns in the same circuit-then-layer order as (a) --- each digit loads almost only its own circuit. \textbf{(c)} On the CIFAR-10 CNN, mean fingerprint per class over the 156 factors, grouped into the ten output circuits; every class puts its largest share on its own circuit. \textbf{(d,~e)} First two principal components of the fingerprints and of the network's own $256$-d penultimate activations, dots colored by class: the $156$-d fingerprint is the more class-separable code (silhouette $0.54$ against $0.44$). \textbf{(f)} Pairwise cosine of the 600 test fingerprints, sorted by class (within-class cosine $0.77$, between-class $0.33$). \textbf{(g)} The same embedding for the pretrained 1000-way SqueezeNet over 8 held-out ImageNet categories: the $253$-d fingerprint separates them all (silhouette $0.51$). \textbf{(h)} The network's own penultimate representation on the same stimuli, its $512$-d classifier input: it separates the categories far less cleanly (silhouette $0.21$ against the fingerprint's $0.51$). \textbf{(i)} The silhouettes of (d,e) and (g,h) side by side --- the fingerprint against the network's own activations, each code at its own native dimension: the fingerprint wins on both datasets, $0.54$ against $0.44$ on CIFAR-10 and $0.51$ against $0.21$ on ImageNet. A dimension-matched comparison under one protocol, with random-projection and last-layer controls, is Fig.~P. The 8x4 MLP's own fingerprint analysis is Fig.~C, the CNN's is Fig.~F, the ImageNet one Fig.~O.

## appendix/figC_mlp_fingerprint_details.pdf  (appendix, full width)

source: `notebooks/01_MLP_8_4_0134.ipynb`, §5b

message: The held-out and far-OOD behaviour of the 8x4 even/odd MLP fingerprint and the well-conditioned NNLS machinery behind it (the embedding and separability panels moved to the cross-model figures P/Q/R).

caption: \textbf{Fingerprint analysis of the $8\times4$ even/odd MLP.} \textbf{(a)} For the six held-out digits (2,\,5--9) the odd-circuit share of the fingerprint tracks the network's own $P(\text{odd})$, not the true parity ($r=0.999$; open circles are the trained digits). \textbf{(b)} Cosine of each fingerprint to its condition's mean: far-OOD collapses onto one fingerprint. \textbf{(c)} Mean fingerprint of every condition (full version of Fig.~4a), the factors grouped by even/odd circuit. \textbf{(d)} NNLS round-trip: projecting the test set onto the fixed factors recovers the NMF's own fingerprint (mean cosine $0.994$, $98.3\%$ above $0.99$; log counts). \textbf{(e)} Pairwise cosine similarity, 100 stimuli per digit --- the blocks behind the silhouettes: $0.89$ by class, $0.56$ by digit.

## (removed) fig5_digit_mlp_fingerprints — merged into figD_digit_mlp_fingerprint_details

The digit-MLP fingerprint result (no space for it as a main figure) is now the first half of the merged appendix figure below.

## appendix/figD_digit_mlp_fingerprint_details.pdf  (appendix, full width) — merged main + details

source: `notebooks/02_MLP_40_20_digits.ipynb` (bundle `nb02_fingerprints`)

message: At ten classes the far-OOD collapse and the well-conditioned NNLS machinery of the digit-MLP fingerprint (the embedding, separability and Fashion-MNIST agreement panels moved to the cross-model figures P/Q/R).

caption: \textbf{Fingerprints in the $40\times20$ digit MLP.} \textbf{(a)} Far-OOD inputs collapse onto a single fingerprint (cosine of each fingerprint to its condition's mean). \textbf{(b)} Mean fingerprint of every condition, grouped by the seven circuits $f_0$--$f_6$. \textbf{(c)} NNLS round-trip: with 125 factors the projection is the loosest of the paper's models (mean cosine $0.919$, min $0.341$, $61\%$ above $0.95$; log counts). \textbf{(d)} Pairwise cosine similarity, 40 stimuli per digit (silhouette $0.56$).

## appendix/figG_vit_circuits.pdf  (appendix, full width)

source: `notebooks/fig04_vit.ipynb`, §3 (bundle exported by `notebooks/04_ViT.ipynb`, §3a)

message: BFT runs on a transformer block exactly as it does on an MLP — attention (W_V), W_O and both FFN layers factorize into the same kind of trace — and it finds the same phenomenology: output-layer circuits that pool visually similar digits along parity, splitting at FFN1 into digit-specific sub-factors. The honest caveat is where the ViT differs: the CLS attention pattern is shared by every factor, so what separates the circuits lives in the value and FFN arbors, not in where the model looks. This is the only ViT circuits figure of the paper.

caption: \textbf{The complete BFT trace of the single-block TinyViT ($d=32$, 2 heads, FFN 64) on MNIST even/odd.} The trace runs backward through B0-FFN2 $\to$ B0-FFN1 $\to$ B0-O $\to$ B0-V, with the attention layer entered through its attention-weighted effective token. \textbf{(a)} Informativity spectra of all 16 nodes, grouped by layer (bar: mean over the nodes of that layer, dots: individual nodes) --- every node is dominated by two or three factors, as in the MLPs. \textbf{(b)} The output layer splits into three circuits: $f_0$ (odd, $0.88$; digits 1,\,7,\,9), $f_1$ (odd, $0.73$; digits 3,\,5) and $f_2$ (even, $0.87$; digits 0,\,6,\,4,\,2,\,8) --- factors pool visually similar digits inside a parity, exactly as in Fig.~3. \textbf{(c)} The 64 FFN units each circuit recruits: distributed and partly shared (mean pairwise cosine $0.35$). \textbf{(d)} At $L_{\mathrm{FFN1}}$ each circuit splits into digit-specific sub-factors; their weighted-average driving input looks like the digit their loading profile names. \textbf{(e)} CLS attention per factor: the mean map (left) and each leaf factor's deviation from it. All 36 factor maps are near-identical (pairwise $\cos\geq0.97$) --- in this ViT the factors are distinguished by the value/FFN arbor, not by where attention points. \textbf{(f)} The eight strongest real stimuli of each output circuit.

## appendix/figH_vit_fingerprints.pdf  (appendix, full width)

source: `notebooks/fig04_vit.ipynb`, §3 (bundle exported by `notebooks/04_ViT.ipynb`, §5a)

message: The fingerprint transfers to the transformer unchanged: a 75-d code whose three blocks are the output circuits, class-structured on MNIST (parity silhouette 0.32, digit 0.11 — weaker than the MLPs, and reported as such), recovered almost exactly by the NNLS projection, embedding as clearly by parity as the ViT's own FFN activations, and showing the same OOD signature as every other model. This is the only ViT fingerprint figure of the paper.

caption: \textbf{BFT fingerprints of the TinyViT.} \textbf{(a)} Mean fingerprint per digit over the 75 factors, grouped into the three output circuits $f_0$--$f_2$ of Fig.~G; the strip on the right sums each block. \textbf{(b)} Pairwise cosine similarity, 30 stimuli per digit (silhouette $0.32$ by parity, $0.11$ by digit --- lower than the MLPs of Figs.~4 and~5). \textbf{(c)} Mean fingerprint of every stimulus condition; projecting held-out stimuli onto the fixed factors reproduces the NMF's own loadings ($r=0.998$--$0.999$). \textbf{(d)} Mean fingerprint similarity between conditions: no OOD condition comes close to the odd fingerprint, while the four far-OOD conditions are nearly interchangeable ($0.85$--$0.97$). \textbf{(e)} First two principal components of the fingerprints, with Fashion-MNIST and far-OOD projected in, beside the same embedding of the ViT's own FFN activations (an independent sample of the same test set --- this model's stimulus order could not be reproduced exactly). \textbf{(f)} Cosine of each fingerprint to its condition's mean: in-distribution stimuli stay spread out, far-OOD inputs collapse onto one fingerprint. \textbf{(g)} Maximum cosine to a trained-class centroid, per condition.

## appendix/figQ_rank_sensitivity.pdf  (appendix, full width) — rank sensitivity, all five models

source: render `figQ_rank_sensitivity` (loads all five `nb09_<exp>_validation` bundles).

message: The factors are a property of the arbor, not of the exact NMF rank. A 2x5 grid of small multiples, one column per model: the top row re-runs the factorization at $K^{*}-1$, $K^{*}$ and $K^{*}+1$ and measures the cosine to the $K^{*}$ factors (they barely move); the bottom row plots the arbor reconstruction $R^2$ against the rank for every layer, marking the rank actually used --- it sits on a plateau, not a cliff. This is the former per-model figI--M panels (e) and (f) lifted out and shown across all five models at once.

caption: \textbf{The factors do not depend on the exact NMF rank, on any model.} One column per model. \textbf{(a)} Re-running the factorization at $K^{*}\pm1$ recovers nearly the same factors (cosine to the $K^{*}$ factors; each dot a node, bar the median). \textbf{(b)} Arbor reconstruction $R^2$ against rank $K$ for every layer (dot: the rank used; line shade: layer depth) --- the chosen rank sits on a plateau.

## appendix/figR_silhouette.pdf  (appendix, full width) — silhouette scores and the weight advantage, all five models

source: render `figR_silhouette` (loads all five `nb09_<exp>_validation` bundles).

message: The BFT fingerprint is the more class-separable code on every model, and the weight term is what buys it. (a) On all five models the fingerprint out-silhouettes the network's dimension-matched activations. (b) The same NMF run on the arbor ($W\!\cdot\!a$) separates classes at least as well as run on the activations alone, so multiplying the weights in earns its place; the advantage is largest on the legible MLPs and shrinks (but never reverses) on the deeper nets.

caption: \textbf{Class separability across all five models.} \textbf{(a)} Cosine silhouette of the BFT fingerprint vs the network's dimension-matched penultimate activations --- the fingerprint is the more separable code on every model. \textbf{(b)} The weight term earns its place: silhouette of the same NMF run on the arbor ($W\!\cdot\!a$) vs on the activations alone. Both read from the five `nb09` validation bundles.

## appendix/figS_ood_signature.pdf  (appendix, full width) — the OOD signature, across models

source: render `figS_ood_signature` (loads all five `nb0N_fingerprints` bundles); collapse and centroid cosines computed on the render side with numpy.

message: The OOD behavior each per-model fingerprint figure (figC/D/F/H/O) shows one model at a time, unified: every model in the paper shows the same signature. In-distribution and near-OOD (real held-out data --- held-out digits, Fashion-MNIST, CIFAR-100) fingerprints stay spread out and land on a trained class; the four synthetic far-OOD conditions collapse onto essentially one fingerprint (mean cosine to their own mean $0.93$--$0.96$ on all five models) and, for the deeper nets, sit off every class. A fingerprint that is both concentrated and far from any centroid is the network flagging "nothing I recognize."

caption: \textbf{The same OOD signature on every model.} \textbf{(a)} How tightly each condition collapses onto a single fingerprint (mean cosine of each fingerprint to its condition mean): in-distribution and near-OOD stay spread ($0.54$--$0.79$), while the synthetic far-OOD conditions collapse ($0.93$--$0.96$) on all five models. \textbf{(b)} Maximum cosine of each condition's mean fingerprint to a trained-class centroid: in-distribution lands on a class, near-OOD (real held-out data) still lands among the classes ($\approx0.77$), and far-OOD sits further off --- most clearly on the ImageNet SqueezeNet ($0.30$). The SqueezeNet ships no near-OOD condition, so its near bars are absent (not zero). Per-model detail is Figs.~C, D, F, H, O.

## (removed) figI–figM per-model validation — restructured into figP/figQ/figR by topic

The five per-model validation figures (one 9-panel `fig_validation` per model) were replaced by three cross-model topic figures, each loading all five `nb09_<exp>_validation` bundles: **figP_faithfulness** (reconstruction controls + NNLS round-trip + NMF-seed stability), **figQ_rank_sensitivity** (the former panels e/f, rank $K^{*}\pm1$ robustness and the rank sweep), and **figR_silhouette** (fingerprint-vs-activation silhouette + the weight-term advantage). The pixel-attribution comparison (former panel i) was dropped. Bundles and `scripts/build_validation_bundles.py` are unchanged; only the render side was restructured, so `notebooks/fig09_validation.ipynb` and the `fig_validation` render function are obsolete.

## fig6_cnn_circuits.pdf  (main text, full width)

source: `notebooks/fig03_cnn_cifar.ipynb`, §3 (bundle exported by `notebooks/03_CNN_CIFAR10.ipynb`, §3)

message: The stimuli that drive the factors make the CIFAR-10 CNN legible across depth — at the output every factor concentrates on one class, one layer back each circuit splits into near-disjoint appearance groups (color, pose, background, and for the bird circuit a group of airplanes), and at conv1 class identity is gone: the four factors are the network's color channels, the same four in every circuit.

caption: \textbf{The stimuli that drive each factor make the CIFAR-10 CNN legible from the logits to the pixels.} Layers are numbered $L_1$ (nearest the pixels) to $L_5$ (the classifier), as in Fig.~2. \textbf{(a)} Each output-layer factor concentrates on a single class: its four strongest stimuli are visually stereotyped and its distribution over all ten classes (bars, common scale) piles onto one class (largest share $0.46$--$0.70$, printed below; the $f$ label sits under the stimuli). \textbf{(b)} Tracing three factors one layer back to $L_4$, each into two sub-factors (node $\propto$ output factor, edges toward the input): the factor un-pools into near-disjoint appearance groups (by color, pose and background). \textbf{(c)} At $L_1$ the four factors of $f_9$ are color channels (bottom strip: RGB input arbor, number below: class purity), the same four in all ten circuits (cosine $0.99$). \textbf{(d)} $\lambda$-weighted class purity falls from $0.60$ at $L_4$ to $0.15$--$0.18$ below it (chance $0.10$). \textbf{(e)} Meanwhile the top sets grow more color-coherent, from a spread near the $0.28$ of random stimuli at $L_4$ down to $0.10$ at $L_1$. Appendix~E gives all sixty $L_4$ sub-factors.

## appendix/figE_cnn_details.pdf  (appendix, full width)

source: `notebooks/fig03_cnn_cifar.ipynb`, §3 (same bundle)

message: Fig. 6 traces three circuits into conv4 with two sub-factors each; the appendix widens that gallery to all ten circuits and, for every conv4 sub-factor, shows its weighted-average stimulus (the factor's prototype) beside six real top stimuli, so each reads as a coherent appearance group rather than a single lucky image. It also gives every node's spectrum and the full traced spine of one circuit to the pixels.

caption: \textbf{The BFT trace of the CIFAR-10 CNN behind Fig.~6, in more detail.} \textbf{(a)} $\lambda$ spectra of all 41 nodes, grouped by layer (bar: mean over the nodes of that layer, dots: individual nodes). \textbf{(b)} Gallery: each output circuit $f_r$ traced back to its two strongest $L_4$ sub-factors $f_k$; every sub-factor is shown as its weighted-average stimulus (framed, the prototype) then six real top stimuli, with all factor labels below the images. Rose marks a sub-factor whose dominant class is not the circuit's. \textbf{(c)} The traced spine of $f_9$ from the classifier ($L_5$) to $L_1$, class purity below each factor and the $L_1$ R/G/B input arbor as a colored strip. Layers are numbered $L_1$ (pixels) to $L_5$ (classifier), as in Fig.~6.

## (removed) fig7_cnn_fingerprints — merged into fig4_fingerprints_main + figF

The CIFAR-10 CNN fingerprint is now the second half of the single main fingerprint figure (`fig4_fingerprints_main`, panels c--f): mean fingerprint per class, the two embeddings, and the class geometry. Its remaining panels — CIFAR-100 agreement and far-OOD collapse — moved to Fig.~F.

## appendix/figF_cnn_fingerprint_details.pdf  (appendix, full width)

source: `notebooks/fig03_cnn_cifar.ipynb`, §3 (same bundle; now also carries the CIFAR-100 and far-OOD panels moved from the old Fig. 7)

message: The fingerprint machinery of the CIFAR-10 CNN behind the main figure — the well-conditioned NNLS round-trip (loosest of the paper's models at 0.94), every condition's mean fingerprint, the far-OOD collapse, the distance of each condition to the classes, and stability across the train/test split.

caption: \textbf{Fingerprint analysis of the CIFAR-10 CNN.} \textbf{(a)} NNLS round-trip: projecting held-out stimuli onto the fixed factors recovers the NMF's own fingerprint (mean cosine $0.937$, min $0.836$), between the $8\times4$ MLP's $0.994$ and the digit MLP's $0.919$. \textbf{(b)} Mean fingerprint of every condition, the full version of the main figure's per-class panel; far-OOD conditions concentrate on a sparse subset of factors. \textbf{(c)} Cosine of each fingerprint to its condition's mean: far-OOD collapses onto one fingerprint. \textbf{(d)} Maximum cosine of each condition to a class centroid. \textbf{(e)} Class centroids on train vs test: diagonal $0.97$ against $0.35$ off-diagonal, all ten match.

## fig8_imagenet_circuits.pdf  (main text, full width)

source: `notebooks/fig05_imagenet.ipynb`, §3 (bundle exported by `notebooks/05_imagenet_cnn.ipynb`, §3)

message: BFT scales unchanged to a pretrained 1000-way SqueezeNet 1.1: traced along its squeeze spine over 8 held-out categories, the stimuli that drive each factor make it legible from the logits to the pixels — every output factor is one category, one fire module back each circuit un-pools into appearance sub-groups within that category, and category purity decays to chance toward the input. This is the only ImageNet circuits main figure.

NOTE the traceback shows f0/f1/f2. Only root factors with a traced sub-tree can appear there, and the root entry of `N_BRANCHES` (5) decides how many exist, so f0--f4 are the candidates; f5--f9 are output factors without children. The conv1 color-factor panel and the spatial-map panel were dropped from the main figure; the spatial maps live in Fig. N(d).

caption: \textbf{The BFT trace of a pretrained 1000-way SqueezeNet 1.1, over 8 held-out ImageNet categories, read from the logits to the pixels.} Layers are numbered $L_1$ (nearest the pixels) to $L_{10}$ (the classifier), as in Fig.~2. \textbf{(a)} Each of the 10 output-layer factors concentrates on a single category: its four strongest val stimuli are visually stereotyped and its distribution over the 8 categories (bars, common scale) piles onto one (largest share $0.52$--$0.74$, printed below; the $f$ label sits under the stimuli). \textbf{(b)} Tracing three circuits one layer back to $L_8$, each into two sub-factors (node $\propto$ output factor, edges toward the input): the circuit un-pools into appearance sub-groups \emph{within} its category --- bear species, airplane views, dog poses --- never across categories. \textbf{(c)} $\lambda$-weighted category purity falls from $0.6$ at the classifier ($L_{10}$) to $0.14$--$0.19$ below $L_8$ (chance $0.12$): the factors stop being about category and become shared low-level features. Appendix~N gives the full trace, the conv1 color factors and the spatial activation maps.

## appendix/figN_imagenet_details.pdf  (appendix, full width)

source: `notebooks/fig05_imagenet.ipynb`, §3 (same bundle)

message: The Fig. 8 trace in full — every node's informativity spectrum, a wider gallery of each circuit's fire8 sub-factors (each shown as its weighted-average stimulus beside real top stimuli), and one circuit followed all the way from the classifier to conv1, where category identity dissolves into shared low-level features.

caption: \textbf{The BFT trace of the ImageNet SqueezeNet behind Fig.~8, in more detail.} \textbf{(a)} $\lambda$ spectra of all 86 nodes, grouped by layer (bar: mean over the nodes of that layer, dots: individual nodes) --- every node is dominated by one to three factors. \textbf{(b)} Gallery: each output circuit $f_r$ traced back to its three strongest $L_8$ sub-factors $f_k$, one circuit per row; every sub-factor is shown as its weighted-average stimulus (framed, the prototype) then its real top stimuli, with all factor labels below the images. \textbf{(c)} One circuit (airplane) traced from the classifier ($L_{10}$) to $L_1$: category purity (below each factor) decays to chance ($0.12$) as the factors become shared, category-agnostic low-level features --- the stereotyped airplanes at the top give way to generic textures. Layers are numbered $L_1$ (pixels) to $L_{10}$ (classifier), as in Fig.~8.

## (removed) fig9_imagenet_fingerprints — merged into fig4_fingerprints_main + figO

The paper keeps a single fingerprint main figure. The ImageNet fingerprint embedding became `fig4_fingerprints_main` row 3 (g), beside the activation slot (h) and the fingerprint-vs-activations silhouette for both datasets (i). Its other panels were already in Fig.~O or moved there: the per-category means are Fig.~O(b), the class geometry Fig.~O(c) (which replaced the far-OOD plane — it duplicated Fig.~4g and collided with the bear class color), far-OOD in O(d)/(e)/(g), the NNLS round-trip in O(a).

## appendix/figO_imagenet_fingerprint_details.pdf  (appendix, full width)

source: `notebooks/fig05_imagenet.ipynb`, §5 (same bundle)

message: The fingerprint machinery behind the ImageNet row of Fig. 4 — every condition's mean fingerprint, the block class geometry, the far-OOD collapse, how far the far-OOD conditions sit from any category, and the code's reproduction across an independent val split.

caption: \textbf{Fingerprint analysis of the ImageNet SqueezeNet.} \textbf{(a)} Mean fingerprint of every condition (8 categories, 4 far-OOD), grouped into the output circuits; the far-OOD conditions concentrate on a sparse subset of factors. \textbf{(b)} Pairwise cosine of the 544 val fingerprints, sorted by category --- the block diagonal behind the silhouette of Fig.~4g (within-category cosine $0.77$). \textbf{(c)} Cosine of each fingerprint to its condition's mean: far-OOD collapses onto one fingerprint. \textbf{(d)} Maximum cosine of each condition to a category centroid: the synthetic far-OOD conditions sit far from every category ($0.10$--$0.15$), inverted images less so ($0.73$). \textbf{(e)} Pairwise cosine between the fingerprints of two independent val splits, sorted by category: within-category $0.84$ against $0.27$ across, so the code is a property of the category, not of the sample.

## appendix/figP_faithfulness.pdf  (appendix, full width) — faithfulness, all five models

source: render `figP_faithfulness` (loads all five `nb09_<exp>_validation` bundles).

message: The decomposition is faithful on every model. Three cross-model panels: (a) the causal reconstruction $R^2$ sits between a random-rank floor and the exact-injection ceiling; (b) the NNLS projection round-trip re-explains held-out stimuli; (c) the factors survive reseeding, above the 0.85 stability gate. The two honest gaps are shown as ``n/a'', not hidden: the pretrained SqueezeNet has no causal reconstruction (spatial pooling is not invertible) and the ViT's attention nodes have no NNLS round-trip.

caption: \textbf{The decomposition is faithful on every model.} \textbf{(a)} Causal reconstruction $R^2$ (median over reconstructable nodes), bracketed by a random-rank floor and the exact-injection ceiling (tick marks). \textbf{(b)} NNLS round-trip fidelity (median cosine of held-out stimuli re-projected onto the fixed factors). \textbf{(c)} Stability of the factors across NMF seeds (mean matched cosine), all above the $0.85$ gate. ``n/a'' marks a metric a model genuinely cannot supply (no causal reconstruction for the pretrained SqueezeNet, no round-trip for the ViT's attention nodes).
