# Figure log

One entry per figure: key message + final caption. Keep in sync with edits.

Venue: AAAI (tueplots `aaai2024` geometry: 6.975 in text width, 3.3 in column width, 9 pt body, Times via `\usepackage{times}`). Include with `\includegraphics{...}` and NO `width=` — figures are produced at final size.

## fig2_mlp_circuits.pdf  (main text, Fig. 2, full width)

source: `notebooks/01_MLP_8_4_0134.ipynb`, last code cell but one

message: BFT splits the parity net into two class circuits that barely share hidden units and push-pull at the output, and each circuit refines into digit-selective sub-circuits toward the pixels — the digit split just happens at different depths in the two circuits.

caption: \textbf{BFT splits the $784\to8\to4\to2$ even/odd MLP into two near-disjoint class circuits that refine into digit-selective sub-circuits toward the input.} \textbf{(a)} Both circuits in one graph (node area $\propto$ loading, unit order sorted by owning circuit; solid excitatory, dashed inhibitory): they claim different $L_1$ units (support cosine $0.27$ vs $0.59$ for a within-circuit shuffle, $p=0.015$) and each drives its own logit while suppressing the other. \textbf{(b)} Per-digit loading of the $L_3$ and $L_2$ factors (digits ordered 0,\,4,\,1,\,3; blue even, red odd). The even circuit already splits into a ``4'' and a ``0'' factor at $L_2$; the odd circuit does not split until $L_1$. \textbf{(c)} Digit purity of every factor (dot area $\propto\lambda$ share, line: $\lambda$-weighted mean) rises monotonically toward the input, $0.49\to0.63\to0.82$ against a chance level of $0.25$. \textbf{(d,~e)} Pixel arbor and digit loading of the four $L_1$ factors below each circuit's $L_2\,k_0$ (bold: the digit the factor loads most): 4- and 0-detectors on the even side, 1- and 3-detectors on the odd side.

## fig3_digit_mlp_circuits.pdf  (main text, Fig. 3, full width)

source: `notebooks/02_MLP_40_20_digits.ipynb`, last code cell but one

message: The ten-class net does not factorize into digits at the output — it factorizes into seven distributed, heavily overlapping circuits that each pool several digits, and tracing a circuit backward un-pools it into digit-selective factors.

caption: \textbf{In the $784\to40\to20\to10$ MLP the output layer factorizes into seven circuits that each pool several digits; tracing a circuit back to the pixels un-pools it again.} \textbf{(a)} Per-digit loading of the seven output-layer factors — the largest digit share is only $0.28$--$0.54$, so no output factor is a digit detector. \textbf{(b)} Digit purity along each circuit (grey: every factor of the node; thin lines: each circuit's purest factor; thick: their mean): the purest factor rises $0.41\to0.75\to0.77$ while the spread over all factors stays wide. $20$ of the $21$ pooled digits (three per circuit) get an $L_1$ factor of their own, against $6$ when the same digits are scored with another circuit's factors. \textbf{(c)} The circuits are distributed, unlike Fig.~2: $25$ of $40$ $L_1$ units per circuit and a pairwise support overlap of $0.77$, \emph{above} the $0.63$ of a shuffle null. \textbf{(d)} For each circuit, the pixel arbor of the $L_1$ factor that best detects its first, second and third pooled digit (bold: that digit; $k$: which of the circuit's factors) --- e.g.\ $f_3$ pools 2,\,0,\,7 and supplies a 2-, a 0- and a 7-detector.

## appendix/figB_digit_mlp_details.pdf  (appendix, full width)

source: `notebooks/02_MLP_40_20_digits.ipynb`, last code cell

message: The Fig. 3 decomposition rests on a graded spectrum per node, circuits that span the whole network, and the digit profile of every L1 factor — not only the three the main figure has room for.

caption: \textbf{Decomposition details for the $40\times20$ digit MLP.} \textbf{(a)} $\lambda$ spectra of the output node and of the $L_1$ node of every circuit. \textbf{(b)} Each circuit through the full network; of its $\approx1000$ edges only each unit's strongest input is drawn, or the graph is a hairball. \textbf{(c)} Digit profile of \emph{every} $L_1$ factor, the full version of the evidence summarized in Fig.~3(b). \textbf{(d)} Weighted-average driving input of exactly the factors Fig.~3(d) draws (dark = high): the average stimulus of each is a legible instance of the pooled digit it was selected for.

## appendix/figA_mlp_details.pdf  (appendix, full width)

source: `notebooks/01_MLP_8_4_0134.ipynb`, last code cell

message: The Fig. 2 decomposition is well conditioned: a graded spectrum per node, an output layer wired as clean push-pull, hidden-unit supports that barely touch, sign-mirrored connection maps, and driving stimuli that confirm what each L1 factor detects.

caption: \textbf{Decomposition details for the $8\times4$ even/odd MLP.} \textbf{(a)} $\lambda$ spectrum of every node of the trace. \textbf{(b)} Each output-layer factor's arbor mass, split into what it excites and what it inhibits: the even factor drives the even logit and suppresses the odd one, and vice versa, with excitation and inhibition of comparable magnitude. \textbf{(c)} How the two circuits divide the eight $L_1$ units (support cosine $0.27$ against a shuffle null of $0.59\pm0.15$): units~1--4 are even-only, units~0 and~6 odd-only, and only units~5 and~7 carry both. \textbf{(d)} Connection maps (out $\times$ in) at $L_3$ and $L_2$: the inhibitory map mirrors the excitatory one. \textbf{(e,~f)} Weighted-average driving stimulus (dark = high) and inhibitory arbor of every $L_1$ factor; the bold digit is the one the factor loads most.

## fig4_mlp_fingerprints.pdf  (main text, full width)

source: `notebooks/01_MLP_8_4_0134.ipynb`, §5a

message: A stimulus's BFT fingerprint is a compact, class-structured code that is more digit-separable than any raw activation layer of the same network — visible directly in the embedding, where the 13-d fingerprint pulls the four digits apart and the 4-d penultimate activations only separate the two classes — reports the network's own decision on digits the factorization never saw, and collapses to a single point on far-OOD input.

caption: \textbf{The 13-dimensional BFT fingerprint is a class-structured code that reports what the network does --- including on stimuli the factorization never saw.} \textbf{(a)} Mean fingerprint per digit; columns are the tree's factors grouped by circuit (blue even, red odd) and layer. Each digit loads almost only its own circuit, with a distinct $L_1$ profile. \textbf{(b,~c)} First two principal components of the fingerprints and of the network's own penultimate activations, same stimuli, dots colored by digit: the fingerprint separates all four digits (silhouette $0.57$), the $4$-d activations only the two classes ($0.03$). \textbf{(d)} Digit separability against the network's activations: at 13 dimensions the fingerprint has the highest silhouette ($0.56$ against $0.32$ for the 8-d $L_1$ activations and $0.28$ for all 796 activations together) at a $5$-NN accuracy of $0.98$. \textbf{(e)} For the six held-out digits (2,\,5--9) the odd circuit's share of the fingerprint tracks the network's own $P(\text{odd})$, not the true parity ($r=0.999$; open circles are the trained digits). \textbf{(f)} Cosine of each fingerprint to its condition's mean: in-distribution and held-out stimuli stay spread out, far-OOD inputs collapse onto a single fingerprint.

## appendix/figC_mlp_fingerprint_details.pdf  (appendix, full width)

source: `notebooks/01_MLP_8_4_0134.ipynb`, §5b

message: The fingerprint machinery behind Fig. 4 is well conditioned — the NNLS projection recovers the NMF's own fingerprint, every condition's mean fingerprint is readable, the OOD conditions have a visible place in the fingerprint plane, and the digit blocks behind the Fig. 4(d) silhouette are visible.

caption: \textbf{Fingerprint details for the $8\times4$ even/odd MLP.} \textbf{(a)} NNLS round-trip: projecting the test set onto the fixed factors recovers the fingerprint the NMF itself produced (mean cosine $0.994$, $98.3\%$ above $0.99$; counts log-scaled). \textbf{(b)} Mean fingerprint of every condition, the full version of Fig.~4(a); all four far-OOD types load the same single odd-circuit factor. \textbf{(c)} The plane of Fig.~4(b) with the OOD conditions projected into it: the held-out digits spread along the trained geometry, the far-OOD stimuli pile onto one point at its tip. \textbf{(d)} Pairwise cosine similarity, 100 stimuli per digit --- the block structure behind the silhouettes of Fig.~4(d): $0.89$ by class, $0.56$ by digit. \textbf{(e)} Maximum cosine to a trained-digit centroid, per condition.

## fig5_digit_mlp_fingerprints.pdf  (main text, full width)

source: `notebooks/02_MLP_40_20_digits.ipynb`, §5a

message: At ten classes the fingerprint stays class-structured and still reports the network's own decision on Fashion-MNIST, but it is no longer the most separable code — the 20-d penultimate activations are, and the two embeddings show it side by side. Here it buys traceability, not separability.

caption: \textbf{At ten classes the fingerprint stays class-structured and still reports the network's own decision, but it is no longer the most separable code.} \textbf{(a)} Mean fingerprint per digit over the 125 factors, grouped by the seven output-layer circuits $f_0$--$f_6$ of Fig.~3. \textbf{(b,~c)} First two principal components of the fingerprints and of the network's own penultimate activations, same stimuli, dots colored by digit --- here the $20$-d activations are the tighter code (silhouette $0.82$ against $0.55$). \textbf{(d)} Digit separability against the network's activations, the same ordering as the embeddings ($5$-NN $0.98$ against $0.99$); on this model the fingerprint buys interpretability and traceability, not separability. \textbf{(e)} On Fashion-MNIST the nearest in-distribution digit fingerprint agrees with the network's predicted digit ($r=0.90$ over the 100 class$\times$digit cells; $70\%$ per sample against $10\%$ chance) --- including the network's own idiosyncrasy of funnelling six of the ten clothing classes onto digit~0. \textbf{(f)} As in Fig.~4(f), far-OOD inputs collapse onto a single fingerprint.

## appendix/figD_digit_mlp_fingerprint_details.pdf  (appendix, full width)

source: `notebooks/02_MLP_40_20_digits.ipynb`, §5b

message: Details for Fig. 5 — the NNLS round-trip is looser here than in the 8x4 net (125 factors, mean cosine 0.92), every condition's mean fingerprint is readable, and Fashion-MNIST falls inside the digit geometry rather than outside it, in the block similarities and in the embedding alike.

caption: \textbf{Fingerprint details for the $40\times20$ digit MLP.} \textbf{(a)} NNLS round-trip. With 125 factors the projection is a looser fit than in the $8\times4$ net (mean cosine $0.919$, min $0.341$, $61\%$ above $0.95$, against $0.994$ in Fig.~C): the more factors the fingerprint has, the less uniquely the non-negative least squares problem is determined. \textbf{(b)} Mean fingerprint of every condition, the full version of Fig.~5(a); the far-OOD conditions again concentrate on a handful of factors. \textbf{(c)} The plane of Fig.~5(b) with the OOD conditions projected into it --- Fashion-MNIST lands \emph{among} the digits, which is why the fingerprint still names a digit for it. \textbf{(d)} Block cross-similarity of the ten digits and the ten Fashion-MNIST classes, the same finding in cosines. \textbf{(e)} Pairwise cosine similarity, 40 stimuli per digit --- the block structure behind the silhouette of Fig.~5(d). \textbf{(f)} Maximum cosine to a digit centroid, per condition.

## appendix/figG_vit_circuits.pdf  (appendix, full width)

source: `notebooks/fig04_vit.ipynb`, §3 (bundle exported by `notebooks/04_ViT.ipynb`, §3a)

message: BFT runs on a transformer block exactly as it does on an MLP — attention (W_V), W_O and both FFN layers factorize into the same kind of trace — and it finds the same phenomenology: output-layer circuits that pool visually similar digits along parity, splitting at FFN1 into digit-specific sub-factors. The honest caveat is where the ViT differs: the CLS attention pattern is shared by every factor, so what separates the circuits lives in the value and FFN arbors, not in where the model looks. This is the only ViT circuits figure of the paper.

caption: \textbf{The complete BFT trace of the single-block TinyViT ($d=32$, 2 heads, FFN 64) on MNIST even/odd.} The trace runs backward through B0-FFN2 $\to$ B0-FFN1 $\to$ B0-O $\to$ B0-V, with the attention layer entered through its attention-weighted effective token. \textbf{(a)} Informativity spectra of all 16 nodes, grouped by layer (bar: mean over the nodes of that layer, dots: individual nodes) --- every node is dominated by two or three factors, as in the MLPs. \textbf{(b)} The output layer splits into three circuits: $f_0$ (odd, $0.88$; digits 1,\,7,\,9), $f_1$ (odd, $0.73$; digits 3,\,5) and $f_2$ (even, $0.87$; digits 0,\,6,\,4,\,2,\,8) --- factors pool visually similar digits inside a parity, exactly as in Fig.~3. \textbf{(c)} The 64 FFN units each circuit recruits: distributed and partly shared (mean pairwise cosine $0.35$). \textbf{(d)} At $L_{\mathrm{FFN1}}$ each circuit splits into digit-specific sub-factors; their weighted-average driving input looks like the digit their loading profile names. \textbf{(e)} CLS attention per factor: the mean map (left) and each leaf factor's deviation from it. All 36 factor maps are near-identical (pairwise $\cos\geq0.97$) --- in this ViT the factors are distinguished by the value/FFN arbor, not by where attention points. \textbf{(f)} The eight strongest real stimuli of each output circuit.

## appendix/figH_vit_fingerprints.pdf  (appendix, full width)

source: `notebooks/fig04_vit.ipynb`, §3 (bundle exported by `notebooks/04_ViT.ipynb`, §5a)

message: The fingerprint transfers to the transformer unchanged: a 75-d code whose three blocks are the output circuits, class-structured on MNIST (parity silhouette 0.32, digit 0.11 — weaker than the MLPs, and reported as such), recovered almost exactly by the NNLS projection, embedding as clearly by parity as the ViT's own FFN activations, and showing the same OOD signature as every other model. This is the only ViT fingerprint figure of the paper.

caption: \textbf{BFT fingerprints of the TinyViT.} \textbf{(a)} Mean fingerprint per digit over the 75 factors, grouped into the three output circuits $f_0$--$f_2$ of Fig.~G; the strip on the right sums each block. \textbf{(b)} Pairwise cosine similarity, 30 stimuli per digit (silhouette $0.32$ by parity, $0.11$ by digit --- lower than the MLPs of Figs.~4 and~5). \textbf{(c)} Mean fingerprint of every stimulus condition; projecting held-out stimuli onto the fixed factors reproduces the NMF's own loadings ($r=0.998$--$0.999$). \textbf{(d)} Mean fingerprint similarity between conditions: no OOD condition comes close to the odd fingerprint, while the four far-OOD conditions are nearly interchangeable ($0.85$--$0.97$). \textbf{(e)} First two principal components of the fingerprints, with Fashion-MNIST and far-OOD projected in, beside the same embedding of the ViT's own FFN activations (an independent sample of the same test set --- this model's stimulus order could not be reproduced exactly). \textbf{(f)} Cosine of each fingerprint to its condition's mean: in-distribution stimuli stay spread out, far-OOD inputs collapse onto one fingerprint. \textbf{(g)} Maximum cosine to a trained-class centroid, per condition.

## appendix/figI–figM_validation_*.pdf  (appendix, full width, one per model)

source: `notebooks/fig09_validation.ipynb`, §3 (bundles `nb09_<exp>_validation`, re-encoded from the notebook-09 cluster results by `scripts/build_validation_bundles.py`)

message: The method survives its own controls on every model in the paper, and where it does not, the figure says so. Three rows, one claim each — the decomposition reconstructs the network causally, it does not move when the NMF seed or the rank moves, and its fingerprint is a more class-discriminative code than the network's own activations at the same dimensionality. All five figures share one nine-panel layout so a reader can lay them side by side; panels a model cannot supply are dashed empty frames that state the reason rather than being silently dropped.

The five figures are one render function (`fig_validation`) over five bundles. What differs between them is only the data:

- **figI, MLP even/odd** — every panel populated; the best case.
- **figJ, MLP digits** — the honest negative: BFT reconstructs at 0.65 against a 0.90 rank-matched SVD ceiling, and stability sits on the 0.85 gate.
- **figK, CNN/CIFAR-10** — the first conv setting with a causal reconstruction (0.938), but only at the classifier: spatial pooling is not invertible, so conv nodes cannot be re-injected.
- **figL, ViT** — layer-dict mode has no model to re-run, so (a) and (c) are empty and (b) falls back to the FU2 fc-node measurement, where one FFN node reconstructs catastrophically ($R^2=-236$). Left in the figure deliberately.
- **figM, ImageNet/SqueezeNet** — the flagship separability result: 4x the silhouette of the dimension-matched activations, with a random projection ruling out "PCA merely damaged the activations".

caption (per model; substitute the architecture and the numbers of that bundle): \textbf{Validation of the decomposition on \emph{<model>}: is it faithful, is it robust, does it beat the controls?} \textbf{(a)} Each node's pre-activation is replaced by a rank-$R$ reconstruction and the network is re-run. BFT is bracketed by a random rank-$R$ floor and two ceilings it cannot beat by construction (a rank-matched SVD and exact injection); the activation-only NMF is an advantaged control --- it gets $W$ exactly and only approximates an $(N,n_{\mathrm{in}})$ matrix --- so read it as a second ceiling, not a rival. Dots are individual nodes. \textbf{(b)} The same measurement per node, by depth; the bar is the per-layer median. Held-out and in-sample values are equal, so the factors do not overfit the stimuli they were fit on. \textbf{(c)} Projecting held-out stimuli onto the fixed factors with NNLS recovers the fingerprint the NMF itself produced (log counts). \textbf{(d)} Hungarian-matched cosine between the factors of independent NMF seeds, per layer; dotted: the $0.85$ stability gate. \textbf{(e)} The same match when the factorization is re-run at $K^{*}\pm1$: the factors are a property of the arbor, not of the rank. \textbf{(f)} Arbor $R^2$ against rank for every layer (dot: the rank used) --- the choice sits on a plateau, not a cliff. \textbf{(g)} Silhouette (bar) and 5-NN accuracy (number) of the fingerprint against the network's own activations, both reduced to a common dimension by PCA, with a random projection and a shuffled-label null. \textbf{(h)} Every A1 measurement, arbor NMF ($W\!\cdot\!a$) against activation-only NMF: above the diagonal means the weight term earns its place. \textbf{(i)} Class discriminability of the BFT input map against integrated gradients, saliency and input magnitude. Error bars in (b) and (g) are the spread over five retrained models --- the only meaningful one, since \texttt{bft} initializes NMF deterministically with \texttt{nndsvda}.

## fig6_cnn_circuits.pdf  (main text, full width)

source: `notebooks/fig03_cnn_cifar.ipynb`, §3 (bundle exported by `notebooks/03_CNN_CIFAR10.ipynb`, §3)

message: The stimuli that drive the factors make the CIFAR-10 CNN legible across depth — at the output every factor concentrates on one class, one layer back each circuit splits into near-disjoint appearance groups (color, pose, background, and for the bird circuit a group of airplanes), and at conv1 class identity is gone: the four factors are the network's color channels, the same four in every circuit.

caption: \textbf{The stimuli that drive each factor make the CIFAR-10 CNN legible from the logits to the pixels.} \textbf{(a)} Each output-layer factor concentrates on a single class and its four strongest stimuli are visually stereotyped. \textbf{(b)} One layer back, every circuit splits into six sub-factors holding near-disjoint stimulus groups (top sets overlap $0.06$) — by color, pose and background, and for the bird circuit $f_5$ a group of airplanes. \textbf{(c)} At conv1 the four factors are color channels (bottom strip: RGB input arbor), the same four in all ten circuits. \textbf{(d)} Class purity falls from $0.60$ at conv4 to $0.16$--$0.18$ below it (chance $0.10$) while the top sets grow more color-coherent.

## appendix/figE_cnn_details.pdf  (appendix, full width)

source: `notebooks/fig03_cnn_cifar.ipynb`, §3 (same bundle)

message: Fig. 6 shows three circuits and one leaf; the appendix shows the whole trace — every node's spectrum, the class profile of every output and conv4 factor, the strongest stimulus of all sixty conv4 sub-factors, one circuit's complete spine to the pixels, the color arbor of all ten conv1 nodes, and where a factor actually fires in the image.

caption: \textbf{The complete BFT trace of the CIFAR-10 CNN behind Fig.~6.} \textbf{(a)} $\lambda$ spectra of all 71 nodes, grouped by layer. \textbf{(b,~c)} Class profile of every output factor and of every conv4 sub-factor, on one scale. \textbf{(d)} The strongest stimulus of all sixty conv4 sub-factors --- the complete version of Fig.~6(b); rose marks a group whose dominant class is not the circuit's. \textbf{(e)} The traced spine of $f_9$ from the classifier to conv1. \textbf{(f)} The conv1 input arbor of every circuit: the same red/green/blue split each time. \textbf{(g)} Where a factor fires, for a conv4 and a conv1 node.

## fig7_cnn_fingerprints.pdf  (main text, full width)

source: `notebooks/fig03_cnn_cifar.ipynb`, §3 (bundle exported by `notebooks/03_CNN_CIFAR10.ipynb`, §5)

message: The 240-dimensional BFT fingerprint of the CIFAR-10 CNN is a class-structured code — every class puts most of its mass on its own output circuit, and its embedding is as class-separable as the network's own 256-d penultimate activations — that reports the network's own decision on CIFAR-100 and collapses onto a single point on far-OOD input. Weaker than the MLPs' fingerprints (silhouette 0.40, agreement 0.50), and reported as such.

caption: \textbf{The 240-dimensional fingerprint of the CIFAR-10 CNN is a class-structured code that reports what the network does.} \textbf{(a)} Mean fingerprint per class, columns grouped into the ten output circuits of Fig.~6 and each column scaled to its top class; the strip sums each circuit block --- all ten classes put their largest share on their own circuit (framed; $0.18$ against $0.10$ chance). \textbf{(b,~c)} First two principal components of the fingerprints and of the network's own penultimate activations, same stimuli, dots colored by class: the two codes are comparably class-separable (silhouette $0.40$ against $0.44$) at comparable dimension. \textbf{(d)} Pairwise cosine of the 600 test fingerprints, 60 per class (within-class cosine $0.72$). \textbf{(e)} On CIFAR-100 the class named by the nearest fingerprint centroid agrees with the network's own prediction ($r=0.73$ over the 100 CIFAR-100 $\times$ 10 CIFAR-10 cells, $0.50$ per sample against $0.10$ chance). \textbf{(f)} Cosine of each fingerprint to its condition's mean: in-distribution and CIFAR-100 stimuli stay spread out, noise, gray and checkerboard images collapse onto one fingerprint ($\geq0.99$).

## appendix/figF_cnn_fingerprint_details.pdf  (appendix, full width)

source: `notebooks/fig03_cnn_cifar.ipynb`, §3 (same bundle)

message: Details for Fig. 7, including where the method is weakest — with 240 factors the NNLS round-trip is the loosest in the paper (mean 0.86, 5 of 200 stimuli above 0.95) — while the code itself is stable across the train/test split, and CIFAR-100 sits inside the CIFAR-10 geometry while the far-OOD conditions do not.

caption: \textbf{Fingerprint details for the CIFAR-10 CNN.} \textbf{(a)} NNLS round-trip: with 240 factors the projection is the loosest fit of the paper's models (mean cosine $0.863$, against $0.994$ for the $8\times4$ MLP) --- the more factors, the less uniquely the non-negative least squares problem is determined. \textbf{(b)} Mean fingerprint of every condition, the full version of Fig.~7(a); the far-OOD conditions concentrate on a sparse subset of factors. \textbf{(c)} The plane of Fig.~7(b) with the OOD conditions projected into it: CIFAR-100 covers the CIFAR-10 geometry, the far-OOD stimuli sit off to one side. \textbf{(d)} Class centroids computed on the training set against those computed on the test set: diagonal $0.92$ against $0.38$ off-diagonal, and all ten classes match. \textbf{(e)} Cosine of each class centroid to each OOD condition's mean fingerprint. \textbf{(f)} Maximum cosine to a class centroid, per condition. \textbf{(g)} Share of fingerprint mass per layer; the counts in the legend are the factors that layer contributes.
