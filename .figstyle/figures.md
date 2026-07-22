# Figure log

One entry per figure: key message + final caption. Keep in sync with edits.

Venue: AAAI (tueplots `aaai2024` geometry: 6.975 in text width, 3.3 in column width, 9 pt body, Times via `\usepackage{times}`). Include with `\includegraphics{...}` and NO `width=` — figures are produced at final size.

## fig2_mlp_circuits.pdf  (main text, Fig. 2, full width)

source: `notebooks/01_MLP_8_4_0134.ipynb`, last code cell but one

message: BFT decomposes the tiny MLP into two disjoint class circuits that split into digit-specific sub-circuits at layer 1, whose pixel-space arbors look like the digits they detect.

caption: \textbf{BFT decomposes the $784\to8\to4\to2$ even/odd MLP into two disjoint class circuits that split into digit-specific sub-circuits at layer~1.} \textbf{(a,~b)} Traced circuits (node area $\propto$ loading; solid excitatory, dashed inhibitory): disjoint units, mutual suppression. \textbf{(c)} Per-digit loading of the $L_3$ and $L_2$ factors (digits ordered 0,\,4,\,1,\,3; blue even, red odd); the even branch splits into a ``4'' and a ``0'' factor at $L_2$. \textbf{(d,~e)} Pixel arbor and digit loading of each circuit's five $L_1$ factors: 4- vs 0-detectors (even), 1- vs 3-detectors (odd).

## fig3_digit_mlp_circuits.pdf  (main text, Fig. 3, full width)

source: `notebooks/02_MLP_40_20_digits.ipynb`, last code cell but one

message: In the 10-class MLP the output layer factorizes into factors that pool visually similar digits; tracing each factor to layer 1 splits it back into digit-specific sub-circuits. Circuits are distributed and share hidden units — unlike the tiny even/odd net of Fig. 2.

caption: \textbf{In the $784\to40\to20\to10$ MLP the output layer factorizes into seven factors that pool visually similar digits (largest digit share $0.27$--$0.53$); layer~1 splits them apart again.} \textbf{(a)} Per-digit loading of the output factors. \textbf{(b)} $L_1$ units per circuit: $\approx24$ of 40, overlap $0.75$ vs $0.61$ shuffled --- distributed, unlike Fig.~2. \textbf{(c)} Pixel arbors of the three strongest $L_1$ factors (bold: dominant digit): $f_3$ (2,\,0,\,7) yields 2-, 0- and 7-detectors. \textbf{(d)} Digit purity of each output factor (dash) vs its $L_1$ factors (dots); each circuit's best is purer ($0.62$--$0.82$).

## appendix/figB_digit_mlp_details.pdf  (appendix, full width)

source: `notebooks/02_MLP_40_20_digits.ipynb`, last code cell

message: The Fig. 3 decomposition rests on a long informativity tail, circuits that span the whole net, and layer-1 factors whose digit profiles and driving stimuli confirm the sub-class split.

caption: \textbf{BFT decomposition details for the $40\times20$ digit MLP.} \textbf{(a)} Informativity spectra. \textbf{(b)} Each circuit through the full network (only each unit's strongest edge of $\approx1000$ drawn). \textbf{(c)} Digit profile of \emph{every} $L_1$ factor; Fig.~3(c) shows the top three by $\lambda$. \textbf{(d)} Their weighted-average driving input (dark = high).

## appendix/figA_mlp_details.pdf  (appendix, full width)

source: `notebooks/01_MLP_8_4_0134.ipynb`, last code cell

message: The decomposition behind Fig. 2 is well-conditioned and sign-mirrored: a flat L1 spectrum, inhibitory maps that mirror the excitatory ones, and weighted-average stimuli that confirm the sub-class assignment.

caption: \textbf{Additional BFT structure for the $8\times4$ even/odd MLP.} \textbf{(a)} Informativity spectrum per node. \textbf{(b)} Connection maps at $L_3$/$L_2$: inhibitory mirrors excitatory. \textbf{(c)} Pixel receptive field of each $L_1$ unit under the circuit's dominant factor (index, share of its connection mass). \textbf{(d,~e)} Weighted-average input (dark = high) and inhibitory arbor of every $L_1$ factor.

## fig4_mlp_fingerprints.pdf  (main text, full width)

source: `notebooks/01_MLP_8_4_0134.ipynb`, §5a

message: A stimulus's BFT fingerprint — its loadings on every factor of the trace tree — is a compact, class-structured code that reports what the network actually does: it separates the four digits the even/odd net was never trained to distinguish, tracks the network's own decision on six held-out digits (r = 0.998), and collapses to a single point on far-OOD inputs.

caption: \textbf{The 13-dimensional BFT fingerprint is a class-structured code that reports what the network does --- including on stimuli the factorization never saw.} \textbf{(a)} Mean fingerprint per digit; columns are the tree's factors grouped by circuit (blue even, red odd) and layer. Each digit loads almost only its own circuit, with a distinct $L_1$ profile. \textbf{(b)} Pairwise cosine similarity, 100 stimuli per digit: silhouette $0.89$ by class, $0.56$ by digit. \textbf{(c)} For the six held-out digits (2,\,5--9) the odd circuit's share of the fingerprint tracks the network's own $P(\text{odd})$, not the true parity ($r=0.998$; open circles are the trained digits). \textbf{(d)} Cosine of each fingerprint to its condition's mean: in-distribution and held-out stimuli stay spread out, far-OOD inputs collapse onto a single fingerprint (exactly $1.00$ for constant images).

## appendix/figC_mlp_fingerprint_details.pdf  (appendix, full width)

source: `notebooks/01_MLP_8_4_0134.ipynb`, §5b

message: The fingerprint machinery behind Fig. 4 is well conditioned — the NNLS projection recovers the NMF fingerprint almost exactly, every condition's mean fingerprint is readable, and the 13-d code separates the digits better than any raw activation layer.

caption: \textbf{Fingerprint details for the $8\times4$ even/odd MLP.} \textbf{(a)} NNLS round-trip: projecting the test set onto the fixed factors recovers the NMF's own fingerprint (mean cosine $0.994$; counts log-scaled). \textbf{(b)} Mean fingerprint of every condition, the full version of Fig.~4(a); all four far-OOD types load one odd-circuit factor. \textbf{(c)} First two principal components of the fingerprints. \textbf{(d)} Digit separability against activation controls: at 13 dimensions the fingerprint has the highest silhouette, while $k$-NN is matched by the 784-d pixel and 796-d all-activation controls. \textbf{(e)} Maximum cosine to a trained-digit centroid, per condition.

## fig5_digit_mlp_fingerprints.pdf  (main text, full width)

source: `notebooks/02_MLP_40_20_digits.ipynb`, §5a

message: The same fingerprint code holds up at ten classes: a 50-d fingerprint whose blocks follow the output-layer circuits of Fig. 3, and on Fashion-MNIST it names the same digit the network names (r = 0.88, 68 % per-sample agreement against 10 % chance) while far-OOD inputs collapse.

caption: \textbf{At ten classes the fingerprint stays class-structured and still reports the network's own decision.} \textbf{(a)} Mean fingerprint per digit over the 50 factors, grouped by the five output-layer circuits $f_0$--$f_4$ of Fig.~3. \textbf{(b)} Pairwise cosine similarity, 40 stimuli per digit (silhouette $0.53$); the off-diagonal mass follows digit confusability. \textbf{(c)} On Fashion-MNIST the nearest in-distribution digit fingerprint agrees with the network's predicted digit ($r=0.88$ over the 100 class$\times$digit cells; $68\%$ per sample against $10\%$ chance). \textbf{(d)} As in Fig.~4(d), far-OOD inputs collapse onto a single fingerprint.

## appendix/figD_digit_mlp_fingerprint_details.pdf  (appendix, full width)

source: `notebooks/02_MLP_40_20_digits.ipynb`, §5b

message: Details for Fig. 5 — the NNLS round-trip holds, every condition's mean fingerprint is readable, Fashion-MNIST classes fall inside the digit geometry rather than outside it, and, honestly, the penultimate activations are more digit-separable than the fingerprint on this model.

caption: \textbf{Fingerprint details for the $40\times20$ digit MLP.} \textbf{(a)} NNLS round-trip (mean cosine $0.972$). \textbf{(b)} Mean fingerprint of every condition, the full version of Fig.~5(a). \textbf{(c)} Block cross-similarity of the ten digits and the ten Fashion-MNIST classes: the clothing blocks sit \emph{inside} the digit geometry, not outside it. \textbf{(d)} Digit separability against activation controls --- unlike the even/odd net, here the 20-d penultimate activations are the most separable representation (silhouette $0.82$ against $0.53$); on this model the fingerprint buys interpretability, not separability. \textbf{(e)} Maximum cosine to a digit centroid, per condition.
