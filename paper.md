\documentclass{article}

% if you need to pass options to natbib, use, e.g.:
%     \PassOptionsToPackage{numbers, compress}{natbib}
% before loading neurips_2026

% The authors should use one of these tracks.
% Before accepting by the NeurIPS conference, select one of the options below.
% 0. "default" for submission
\usepackage{neurips_2026}
% the "default" option is equal to the "main" option, which is used for the Main Track with double-blind reviewing.
% 1. "main" option is used for the Main Track
%  \usepackage[main]{neurips_2026}
% 2. "position" option is used for the Position Paper Track
%  \usepackage[position]{neurips_2026}
% 3. "eandd" option is used for the Evaluations & Datasets Track
 % \usepackage[eandd]{neurips_2026}
 % if you need to opt-in for a single-blind submission in the E&D track:
 %\usepackage[eandd, nonanonymous]{neurips_2026}
% 4. "creativeai" option is used for the Creative AI Track
%  \usepackage[creativeai]{neurips_2026}
% 5. "sglblindworkshop" option is used for the Workshop with single-blind reviewing
 % \usepackage[sglblindworkshop]{neurips_2026}
% 6. "dblblindworkshop" option is used for the Workshop with double-blind reviewing
%  \usepackage[dblblindworkshop]{neurips_2026}

% After being accepted, the authors should add "final" behind the track to compile a camera-ready version.
% 1. Main Track
 % \usepackage[main, final]{neurips_2026}
% 2. Position Paper Track
%  \usepackage[position, final]{neurips_2026}
% 3. Evaluations & Datasets Track
 % \usepackage[eandd, final]{neurips_2026}
% 4. Creative AI Track
%  \usepackage[creativeai, final]{neurips_2026}
% 5. Workshop with single-blind reviewing
%  \usepackage[sglblindworkshop, final]{neurips_2026}
% 6. Workshop with double-blind reviewing
%  \usepackage[dblblindworkshop, final]{neurips_2026}
% Note. For the workshop paper template, both \title{} and \workshoptitle{} are required, with the former indicating the paper title shown in the title and the latter indicating the workshop title displayed in the footnote.
% For workshops (5., 6.), the authors should add the name of the workshop, "\workshoptitle" command is used to set the workshop title.
% \workshoptitle{WORKSHOP TITLE}

% "preprint" option is used for arXiv or other preprint submissions
%\usepackage[preprint]{neurips_2026}

% to avoid loading the natbib package, add option nonatbib:
%    \usepackage[nonatbib]{neurips_2026}

\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{hyperref}       % hyperlinks
\usepackage{url}            % simple URL typesetting
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{xcolor}         % colors
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amsthm, amssymb}
\usepackage{geometry}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{cleveref}
\usepackage{algorithm}
\usepackage{algpseudocode}
\bibpunct{(}{)}{;}{a}{,}{,}
\usepackage[normalem]{ulem} % strikethrough

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{example}[theorem]{Example}
\newtheorem{remark}[theorem]{Remark}

\newcommand{\R}{\mathbb{R}}
\newcommand{\St}{\mathrm{St}}
\newcommand{\E}{\mathcal{E}}
\newcommand{\F}{\mathcal{F}}
\newcommand{\D}{\mathcal{D}}


\title{Beyond Activations: Automated Circuit Discovery with Joint Weight-Activation Factorization}


% The \author macro works with any number of authors. There are two commands
% used to separate the names and addresses of multiple authors: \And and \AND.
%
% Using \And between authors leaves it to LaTeX to determine where to break the
% lines. Using \AND forces a line break at that point. So, if LaTeX puts 3 of 4
% authors names on the first line, and the last on the second line, try using
% \AND instead of \And before the third author name.


\author{
}


\begin{document}


\maketitle


\begin{abstract}
Understanding which neurons and weights drive a classification decision is a central challenge in interpretability. Existing attribution methods assign importance to input features but ignore the multi-layer weight structure that produced them, while mechanistic interpretability methods rely on activation only rather than weights. We introduce the \textbf{Backward Factor Trace (BFT)}, a method that decomposes a trained network's computation into interpretable \emph{circuits}: sparse, class-specific subnetworks that collectively explain a class decision from the output layer back to the input pixels. At each layer we form a \emph{joint arbor matrix} — the outer product of weight vectors and input activations over a stimulus population — and factorize it with Non-negative Matrix Factorization (NMF). The resulting factors reveal co-active weight-neuron combinations for different stimuli. BFT propagetes these factors backward as a focusing lens on the preceding layers, yielding connected computational graphs. The method requires no reference input, no gradient computation, and no architectural modification. We validate it on MNIST even/odd and digit classification with small MLPs, demonstrate NMF factor stability across random seeds and show causal circuit specificity via targeted ablation. Preliminary results on a CIFAR-10 CNN and a tiny Vision Transformer show that the algorithm generalizes beyond MLPs.
We further introduce \emph{factor fingerprints} — per-stimulus concatenations of NMF loadings across the full circuit tree — which provide a class-discriminative stimulus embedding that outperforms PCA on raw activations across all tested architectures.
\end{abstract}


\section{Introduction}
\label{sec:introduction}

Modern neural networks achieve remarkable accuracy across vision, language, and scientific domains, yet their internal computations remain largely opaque. This opacity limits trust in high-stakes applications and impedes systematic improvement: when a model fails, we cannot easily diagnose \emph{why}. The field of mechanistic interpretability aims to fill this gap by explaining not just \emph{what} a network predicts, but \emph{which} internal computations produced that prediction.

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figs/Overview_bft.pdf}
    \caption{Preliminary BFT overview}
    \label{fig:placeholder}
\end{figure}

A natural unit of explanation is the \emph{circuit}: a sparse subgraph of neurons and weights that implements a specific computation~\citep{olahZoomIntroductionCircuits2020}. Identifying circuits is appealing because they are both human-interpretable and causally testable — ablating a circuit should disrupt the computation it implements. However, existing approaches to circuit discovery either require costly manual0inspection~\citep{olahZoomIntroductionCircuits2020, Elhage2021mathematicalGoogleSearch}, or rely on input-space attribution methods that identify relevant \emph{features} but not the internal \emph{weight structure} responsible for them.

\textbf{Related Work} Input attribution methods such as saliency maps~\citep{simonyanDeepConvolutionalNetworks2013}, integrated gradients~\citep{sundararajanAxiomaticAttributionDeep2017}, and SHAP~\citep{lundbergUnifiedApproachInterpreting2017a} produce pixel-space heatmaps but do not expose multi-layer circuits. Activation-space methods — including linear probes~\citep{alain2016understanding} and Sparse Autoencoders~\citep{cunningham2023sparse} — decompose representations layer by layer but treat each layer in isolation, ignoring the weights that connect them. Transcoders~\citep{lindsey2025transcoders} partially bridge this gap but require training an auxiliary model and a predefined dictionary. None of these methods answers the fundamental question at each neuron: not merely \emph{did this neuron fire}, but \emph{via which upstream weights}?

We address this gap by building on a simple observation: in any linear layer, neuron $i$ computes $y_i = \sum_j W_{ij} a_j$, so the element-wise product $W_{ij} \cdot a_j$ records precisely which upstream input $j$ contributed to neuron $i$, and by how much. Stacking these \emph{weight-activation products} over all neurons and all input samples produces a joint matrix whose non-negative factorization (NMF) reveals recurring co-activation patterns — groups of upstream inputs and downstream neurons that consistently work together to process a class of stimuli. We call the resulting per-neuron vector a \emph{synaptic arbor}, and the method that traces these arbors backward through the network the \textbf{Backward Factor Trace}.0Crucially, the method requires no reference input, no gradient computation, and no architectural modification; it operates entirely on the weights and activations already present in a trained model.

Our main contributions are:

\begin{itemize}
    \item \textbf{The Backward Factor Trace (BFT)}: an automatic, layer-by-layer algorithm that decomposes a network's computation into interpretable \emph{circuits} — sparse, class-specific subnetworks — by factorizing joint weight-activation arbor matrices with NMF and propagating selectivity-weighted stimulus importance backward from output to input.

    \item \textbf{Stability}: NMF components are highly reproducible across random seeds (cosine similarity $0.938 \pm 0.083$ at L1 after Hungarian matching), demonstrating that the discovered circuits reflect stable structure in the network rather than NMF initialization artifacts.

    \item \textbf{Causal specificity}: ablating the top-$10\%$ neurons identified by the trace drops target-class accuracy by $0.848 \pm 0.181$, compared to $0.505$ for bystander classes ($p < 0.0001$, Wilcoxon), confirming the traced circuits are selectively necessary for their class.

    \item \textbf{Attribution quality}: pixel-space attributions derived from the trace achieve a class discriminability score of $0.658$, outperforming integrated gradients ($0.422$), weight magnitude ($0.378$), and saliency maps ($0.103$).

    \item \textbf{Generalization}: preliminary experiments on a CIFAR-10 CNN and a tiny Vision Transformer demonstrate that the algorithm extends beyond MLPs, recovering within-class sub-circuits (truck vs.\ fire engine; left- vs.\ right-facing horses) and even/odd circuits through Transformer FFN layers.

    \item \textbf{Factor fingerprints}: concatenating a stimulus's NMF loadings over all BFT tree nodes yields a \emph{factor fingerprint} that encodes the full hierarchical circuit activation. MDS on factor fingerprint distances reveals cleaner class structure than PCA on raw network activations, consistent across all tested architectures.
\end{itemize}

\section{Related Work}

\paragraph{NMF for neural analysis.}
\citet{leeLearningPartsObjects1999} introduced NMF for parts-based image decomposition. We apply NMF to the joint matrix of weight-activation products rather than activation matrices alone, directly coupling the weight structure to the stimulus population.

\section{Method}

\subsection{Setup and Notation}

Let $f$ be an $L$-layer MLP with weight matrices $W_l \in \R^{d_l \times d_{l-1}}$ and element-wise sigmoid activations (no bias terms affect the analysis).
For a test set of $N$ correctly-classified [TODO use all samples/wrong samples to find errors in additional analyses] samples, we collect the \emph{input activation} to each linear layer:
$A_l \in \R^{N \times d_{l-1}}$, where $A_l[s,:] = \mathbf{a}_s^l$ is the pre-linear input for sample $s$ at layer $l$.
The goal is to find, for each class $c$, a \emph{circuit} — a sparse subset of neurons and weights across layers — that causally explains the decision $\hat{y} = c$ [TODO Maybe make the framing more general than classification?].

\subsection{Joint Arbor Matrix}
\label{sec:joint-arbor}

Fix a layer $l$ with weight $W \in \R^{d_l \times d_{l-1}}$ and inputs $A \in \R^{N \times d_{l-1}}$.
Let $\mathbf{sw} \in \R^N_{\geq 0}$ be per-sample \emph{stimulus weights} (initially uniform; derived from the layer above in subsequent layers via the selectivity weighting described in \cref{sec:backward}).

\textbf{Step 1: L2-normalize input activations.}
\begin{equation}
  \hat{a}_s^l = \frac{\mathbf{a}_s^l}{\|\mathbf{a}_s^l\| + \varepsilon}.
\end{equation}
This decouples \emph{direction} (which neurons are engaged) from \emph{magnitude} (how strongly), so that the joint arbor captures co-activation patterns regardless of overall activation scale.
The activation magnitudes are absorbed into the stimulus weights.

\textbf{Step 2: Per-neuron arbors.}
For output neuron $i$:
\begin{equation}
  J_s^{(i)}[j] = \mathrm{sw}[s] \cdot W[i,j] \cdot \hat{a}_s^l[j], \quad j = 1, \ldots, d_{l-1}.
\end{equation}

Here, $\mathrm{sw}[s] \geq 0$ is the \emph{stimulus weight} of sample $s$, encoding how relevant that sample is to the circuit being traced.
It is initialized to $1$ (uniform) at the output layer and propagated backward via the selectivity weighting in \cref{sec:backward}.

\textbf{Step 3: Joint arbor matrix.}
Concatenate all neurons' arbors horizontally:
\begin{equation}
  \mathbf{J}_l \in \R^{N \times (d_l \cdot d_{l-1})}, \quad \mathbf{J}_l[s,\; (i-1)d_{l-1}+j] = J_s^{(i)}[j].
\end{equation}

\textbf{Step 4: Sign separation.}
Because NMF requires non-negative inputs, we factor positive (excitatory) and negative (inhibitory) contributions separately:
$\mathbf{J}^+ = \max(\mathbf{J}_l, 0)$ and $\mathbf{J}^- = \max(-\mathbf{J}_l, 0)$.
This preserves inhibitory circuits rather than discarding them.
We trace only the excitatory component $\mathbf{J}^+$ backward.
Because Sigmoid activations are strictly positive, the L2-normalized inputs satisfy $\hat{a}_s^l[j] \geq 0$ for all $s, j$, so the sign of each arbor entry is determined entirely by the weight $W[i,j]$: positive weights yield excitatory entries, negative weights yield inhibitory ones.
Crucially, negative weights \emph{suppress} certain input patterns rather than propagating through them — an inhibitory circuit fires more strongly when its preferred input is \emph{absent}.
The selectivity-based backward propagation (see \cref{sec:backward}) correctly emphasises samples that preferentially activate the excitatory component; applying it to inhibitory factors would invert the selection logic.
Inhibitory factors are therefore computed per-layer as complementary diagnostics that reveal which input patterns are suppressed at each step, without being propagated backward.

\subsection{NMF Factorization and Rank Selection}
\label{sec:nmf}

Apply NMF to $\mathbf{J}^+$:
\begin{equation}
  \mathbf{J}^+ \approx W_\mathrm{img} \cdot H_\mathrm{neu}^\top, \quad W_\mathrm{img} \in \R^{N \times K},\; H_\mathrm{neu} \in \R^{d_l d_{l-1} \times K}.
\end{equation}
$W_\mathrm{img}[:,k]$ (the \emph{stimulus factor}) gives each sample's loading on circuit $k$; $H_\mathrm{neu}[:,k]$ (the \emph{neural factor}) gives each weight-input pair's contribution.

\textbf{Automatic rank selection.}
Rather than specifying $K$ by hand, we fit NMF at an upper-bound rank $k_\mathrm{max}$, normalize each component to unit $\ell^2$ norm and sort by $\lambda_k = \|w_k\|\cdot\|h_k\|$ (component informativity).
The effective rank $K^*$ is chosen by the \emph{structural\_recon} method: find the first consecutive-ratio drop $\lambda_k / \lambda_{k+1} \geq 1.5$ (fraction heuristic) giving $k_\text{frac}$, and the smallest $K$ where the relative Frobenius reconstruction error $\|X - X_K\|_F / \|X\|_F \leq \varepsilon_r$ (reconstruction floor) giving $k_\text{recon}$; then $K^* = \max(k_\text{frac},\, k_\text{recon})$ with default $\varepsilon_r = 0.20$.
This avoids both over-splitting weak components and discarding meaningful ones.

\subsection{Backward Propagation}
\label{sec:backward}

The top stimulus factor $W_\mathrm{img}[:,0]$ identifies which samples most activate circuit $0$.
To trace this circuit into layer $l-1$, we compute:

\textbf{New stimulus weights.}
Let $\tilde{w}_{s,k} = W_\mathrm{img}[s,k] \cdot \lambda_k$ be the lambda-weighted loading of stimulus $s$ on factor $k$.
The stimulus weight passed to layer $l-1$ when tracing factor $k^*$ is its \emph{selectivity}:
\begin{equation}
  \mathrm{sw}^{l-1}[s] = \frac{\tilde{w}_{s,k^*}}{\displaystyle\sum_{k'} \tilde{w}_{s,k'} + \varepsilon}.
\end{equation}
This rewards stimuli that are \emph{selectively} active in circuit $k^*$ relative to all other circuits at the current layer, not just stimuli with high absolute loading.
The normalization ensures $\mathrm{sw} \in [0,1]$ and is the key coupling: the circuit at layer $l$ tells layer $l-1$ which stimuli to factorize.
Connected pathways are enforced by this selectivity: only samples that preferentially flow through circuit $k^*$ at layer $l$ receive high weight, so the NMF at layer $l-1$ is dominated by samples that actually activate the identified circuit — ensuring the traced path is coherent from output back to input.

Initialization: $\mathrm{sw} = \mathbf{1}$ (uniform) at the output layer (layer $L$).

\subsection{Projecting New Stimuli via Backward-Weighted NNLS}
\label{sec:nnls-projection}

Once a BFT tree is fixed on a training population, we can represent any new stimulus $s'$ in the same factor space without refitting NMF.
The key insight is that the neural factors $H_\mathrm{neu}^l$ at an inner layer were trained on a \emph{stimulus-weighted} joint arbor; projecting a new stimulus with an \emph{unweighted} arbor places it in a different space and yields poor inner-layer reconstructions.
The fix is to mirror the BFT backward pass during projection.

\textbf{Root node (output layer).}
For new stimulus $s'$, form the unweighted joint arbor $\hat{\mathbf{J}}_L[s',:] = J_{s'}^{(i)}[j]$ with uniform $\mathrm{sw}[s'] = 1$ (matching the original BFT initialization) and solve per-row NNLS against the stored $H_\mathrm{neu}^L$:
\begin{equation}
  \hat{w}^L_{s'} = \arg\min_{x \geq 0} \bigl\| H_\mathrm{neu}^L\, x - \hat{\mathbf{J}}_L[s',:]\bigr\|_2^2.
\end{equation}

\textbf{Propagating weights to child nodes.}
Given $\hat{W}_\mathrm{img}^L$ and stored $\boldsymbol{\lambda}^L$, compute the stimulus weight for factor $k$ exactly as in \cref{sec:backward}:
\begin{equation}
  \hat{\mathrm{sw}}^{L-1}_k[s'] = \frac{\hat{w}^L_{s',k}\,\lambda^L_k}{\displaystyle\sum_{k'} \hat{w}^L_{s',k'}\,\lambda^L_{k'} + \varepsilon}.
\end{equation}

\textbf{Inner layers.}
For each child node at layer $l-1$ reached by following factor $k$, build the weighted joint arbor
$\hat{\mathbf{J}}_{l-1}^+[s',:] = \mathrm{clip}\bigl(\hat{\mathrm{sw}}^{l-1}_k[s'] \cdot J_{s'}^{\mathrm{raw}}, 0\bigr)$
and solve NNLS against $H_\mathrm{neu}^{l-1}$; then propagate to the next child using the same selectivity formula.

This backward pass ensures that the NNLS at every node operates on a joint arbor in the same weighted space as the NMF factors stored there, removing the mismatch that caused the round-trip cosine similarity to degrade from ${\approx}1.0$ at the root to ${\approx}0.47$ at inner nodes.
The public API is unchanged: \texttt{project\_stimuli\_onto\_tree(root, new\_layer\_inputs)} performs the full backward-weighted NNLS automatically.

\begin{algorithm}[t]
\caption{\textsc{BFT} — Backward Factor Trace}
\label{alg:bft}
\begin{algorithmic}[1]
\Require Model weights $\{W_l\}$, layer inputs $\{A_l\}$, branches $\{B_l\}$, rank bound $k_\mathrm{max}$
\State \textbf{function} \textsc{TraceNode}$(l,\; \mathrm{sw},\; \mathrm{path})$
\State $\quad$ Compute $\mathbf{J}_l$ from $W_l$, $A_l$, $\mathrm{sw}$ \hfill\Comment{\cref{sec:joint-arbor}}
\State $\quad$ $(W_\mathrm{img}, H_\mathrm{neu}, \boldsymbol{\lambda}) \leftarrow \textsc{AutoNMF}(\mathbf{J}_l^+,\; k_\mathrm{max})$ \hfill\Comment{\cref{sec:nmf}}
\State $\quad$ \textbf{if} $l = 1$: \textbf{return} leaf node
\State $\quad$ \textbf{for} $k = 0, \ldots, B_l - 1$ \textbf{do}
\State $\quad\quad$ Compute $\mathrm{sw}^{l-1}_k$ from $W_\mathrm{img}$, $\boldsymbol{\lambda}$, factor index $k$ \hfill\Comment{\cref{sec:backward}}
\State $\quad\quad$ \textsc{TraceNode}$(l-1,\; \mathrm{sw}^{l-1}_k,\; \mathrm{path} \mathbin\| [k])$
\State Initialize: $\mathrm{sw}^L = \mathbf{1}$
\State \textbf{return} \textsc{TraceNode}$(L,\; \mathrm{sw}^L,\; [])$
\end{algorithmic}
\end{algorithm}

\begin{figure}
    \centering
    \begin{tabular}{c}
          \includegraphics[width=0.9\linewidth]{figs/02_factor_trace/mnist_even_odd_mlp_8_4_0134/factor_viz/L3_root_positive.pdf} \\
          (a) MNIST Even/Odd last layer positive factors \\
          \includegraphics[width=0.9\linewidth]{figs/02_factor_trace/mnist_even_odd_mlp_8_4_0134/factor_viz/L3_root_inhibitory.pdf}\\
          (b) MNIST Even/Odd last layer inhibitory factors 
    \end{tabular}
    \caption{The final layer is nicely factorized into one factor for even and odd digits each, showing inverse connection patterns also found in the complementary inhibitory factors. Weighted-average stimulus images and per-class loading distributions confirm clean class separation at this layer.}
    \label{fig:even-odd-last-layer}
\end{figure}

\begin{figure}
    \centering
    \begin{tabular}{c}
          \includegraphics[width=0.9\linewidth]{figs/02_factor_trace/mnist_even_odd_mlp_8_4_0134/factor_viz/L2_F0_positive.pdf} \\
          (a) MNIST Even/Odd middle layer even digit factor \\
          \includegraphics[width=0.9\linewidth]{figs/02_factor_trace/mnist_even_odd_mlp_8_4_0134/factor_viz/L2_F1_positive.pdf}\\
          (b) MNIST Even/Odd middle layer odd digit factor 
    \end{tabular}
    \caption{Tracing the even/odd paths to the middle layer yields only one factor each, again with inverse connection patterns. This suggests the bulk of the computation is performed in layer 1. Interestingly, the odd circuit is a lot sparser, only using one layer 1 and layer 2 neuron each.}
    \label{fig:even-odd-middle-layer}
\end{figure}
\newpage
\section{Results}

\subsection{MNIST Even/Odd: 8$\times$4 MLP}
\label{sec:even-odd}

\textbf{Setup.}
A 3-layer MLP (784 $\to$ 8 $\to$ 4 $\to$ 2, Sigmoid) trained on even/odd classification of digits $\{0,1,3,4\}$.
This minimal architecture provides the clearest test case: 8 neurons in L1, 4 in L2, a binary output, and only 4 digit classes.
\textsc{BFT} is run with $B = [10, 2, 2]$ (10 branches at L1, 2 at L2 and L3).

\textbf{Output layer.}
The last layer's joint arbor factorizes cleanly into exactly two factors — one for even digits, one for odd (\cref{fig:even-odd-last-layer}).
The excitatory and inhibitory components mirror each other: the even circuit excites the even output neuron while suppressing the odd one, and vice versa, confirming the binary decision is encoded as a complementary excitatory/inhibitory pair.
\newpage
\textbf{Middle layer.}
Tracing each branch back to L2 yields a single dominant factor per class (\cref{fig:even-odd-middle-layer}), again with inverse excitatory/inhibitory structure.
Notably, the odd circuit is strikingly sparse, engaging only one L2 and one L1 neuron, suggesting the even-vs-odd distinction is largely resolved in L1.

\begin{figure}[h]
    \centering
    \begin{tabular}{c}
          \includegraphics[width=0.7\linewidth]{figs/02_factor_trace/mnist_even_odd_mlp_8_4_0134/factor_viz/L1_F0-F0_positive.pdf} \\
          (a) MNIST Even/Odd first layer even digit factors \\
          \includegraphics[width=0.7\linewidth]{figs/02_factor_trace/mnist_even_odd_mlp_8_4_0134/factor_viz/L1_F1-F0_positive.pdf}\\
          (b) MNIST Even/Odd first layer odd digit factors 
    \end{tabular}
    \caption{The first layer is more complex and can be decomposed into many factors. The even circuit shows different factors for 4s and 0s, reflecting the need to aggregate over two visually distinct digit shapes within a single class.}
    \label{fig:even-odd-first-layer}
\end{figure}

\begin{figure}[t]
  \centering
  \begin{tabular}{cc}
    \includegraphics[width=0.46\linewidth]{figs/02_factor_trace/mnist_even_odd_mlp_8_4_0134/scaffold_graphs/scaffold_F0-F0.pdf} &
    \includegraphics[width=0.46\linewidth]{figs/02_factor_trace/mnist_even_odd_mlp_8_4_0134/scaffold_graphs/scaffold_F1-F0.pdf} \\
    (a) Even circuit & (b) Odd circuit
  \end{tabular}
  \caption{Scaffold graphs for the even and odd circuits discovered. The odd circuit is very sparse, tracing through a single chain of neurons. Note the inverse inhibitory and excitatory connections in the last layer, reflecting the binary classification structure.}
  \label{fig:even-odd-scaffold}
\end{figure}

\newpage
\textbf{First layer.}
L1 is richer: the even circuit decomposes into multiple factors (\cref{fig:even-odd-first-layer}), distinguishing the two even digit types (0 and 4) via different stroke-pattern detectors.
The top factor captures broad even/odd structure; subsequent factors specialize to individual digit identities.

\newpage
The scaffold graphs (\cref{fig:even-odd-scaffold}) give an integrated view: the even circuit uses a distributed but sparse set of neurons, while the odd circuit is a thin chain.
Both graphs show the inverse sign structure at the output layer.

\newpage
\textbf{Pixel receptive fields.}
Layer-1 receptive fields (\cref{fig:even-odd-rfs}) confirm the hierarchy: high-$\lambda$ factors capture general even/odd patterns, while lower factors isolate features specific to individual digits.




\begin{figure}[t]
  \centering
  \begin{tabular}{cc}
    \includegraphics[width=0.46\linewidth]{figs/02_factor_trace/mnist_even_odd_mlp_8_4_0134/pixel_receptive_fields/L1_F0-F0_positive.pdf} &
    \includegraphics[width=0.46\linewidth]{figs/02_factor_trace/mnist_even_odd_mlp_8_4_0134/pixel_receptive_fields/L1_F1-F0_positive.pdf} \\
    (a) Even-circuit L1 RFs & (b) Odd-circuit L1 RFs
  \end{tabular}
  \caption{Layer-1 pixel receptive fields for the even (left) and odd (right) circuits. Strongest factors (top) show very general patterns, the following factors get more and more specialized, showing single-digit patterns. Each $28\times 28$ panel is the pixel-space receptive field of one L1 neuron participating in the circuit.}
  \label{fig:even-odd-rfs}
\end{figure}

\newpage
\subsection{MNIST Digit Classification: 40$\times$20 MLP}
\label{sec:digit}

\textbf{Setup.}
A 3-layer MLP (784 $\to$ 40 $\to$ 20 $\to$ 10, Sigmoid) trained on all 10 MNIST digits.
\textsc{BFT} uses $B = [1, 1, 10]$: the output layer produces 10 branches (one per digit class), and each branch is traced single-chain through L2 and L1.

\begin{figure}
    \centering

    \includegraphics[width=0.9\linewidth]{figs/02_factor_trace/mnist_digit_mlp_40_20/factor_viz/L3_root_positive.pdf} 

    \caption{For digit classification the last layer factorizes into digit-specific factors (ones, zeros) and factors with some overlap of similar digits (row 2 with fours and nines). Each factor's weighted-average stimulus image and per-class loading distribution are shown.}
    \label{fig:digit-last-layer}
\end{figure}

\textbf{Output layer.}
The last-layer joint arbor factorizes into digit-specific factors (\cref{fig:digit-last-layer}).
Several factors are highly selective to a single digit (e.g., ones and zeros), while others show partial overlap between visually similar digits such as fours and nines — matching the expected structure of a 10-class classifier where confusable classes share representations.

\newpage
\textbf{Tracing the ``ones'' circuit.}
We trace the top output-layer factor (corresponding to the digit ``1'') backward through the network.
At L2 (\cref{fig:digit-middle-layer}) a single dominant factor is strongly selective to ones, with weaker secondary factors showing some activation on similar digits.
At L1 (\cref{fig:digit-first-layer}) the ones circuit decomposes into multiple factors capturing within-class variation — different orientations and stroke styles — which are pooled into the single L2 representation above.


\begin{figure}
    \centering

    \includegraphics[width=0.9\linewidth]{figs/02_factor_trace/mnist_digit_mlp_40_20/factor_viz/L2_F0_positive.pdf}

    \caption{Tracing the Ones factor in the middle layer shows one strong factor selective to ones only, and some other factors also activating on different digits. The dominant factor's neural heatmap shows which L2$\to$L3 weight connections carry the ones signal.}
    \label{fig:digit-middle-layer}
\end{figure}

\begin{figure}[h]
    \centering

    \includegraphics[width=0.9\linewidth]{figs/02_factor_trace/mnist_digit_mlp_40_20/factor_viz/L1_F0-F0_positive.pdf}

    \caption{Traceback of the Ones factor to the first layer shows different factors that capture variations of the digit, such as different orientations. The spread across multiple L1 factors reflects the diversity of stroke styles within the ``1'' class.}
    \label{fig:digit-first-layer}
\end{figure}

\newpage
\textbf{All 10 circuits.}
The scaffold graphs (\cref{fig:digit-scaffold}) show 10 distinct sparse circuits, each engaging a largely non-overlapping subset of the 40 L1 neurons.


\begin{figure}[t]
  \centering
  \begin{tabular}{ccccc}
    \includegraphics[width=0.17\linewidth]{figs/02_factor_trace/mnist_digit_mlp_40_20/scaffold_graphs/scaffold_F0-F0.pdf} &
    \includegraphics[width=0.17\linewidth]{figs/02_factor_trace/mnist_digit_mlp_40_20/scaffold_graphs/scaffold_F1-F0.pdf} &
    \includegraphics[width=0.17\linewidth]{figs/02_factor_trace/mnist_digit_mlp_40_20/scaffold_graphs/scaffold_F2-F0.pdf} &
    \includegraphics[width=0.17\linewidth]{figs/02_factor_trace/mnist_digit_mlp_40_20/scaffold_graphs/scaffold_F3-F0.pdf} &
    \includegraphics[width=0.17\linewidth]{figs/02_factor_trace/mnist_digit_mlp_40_20/scaffold_graphs/scaffold_F4-F0.pdf} \\
    Factor 0 (Digit 1) & Factor 1 (Digits 4) & Factor 2 (Digit 0) & Factor 3 (Digit 6) & Factor 4 \\[4pt]
    \includegraphics[width=0.17\linewidth]{figs/02_factor_trace/mnist_digit_mlp_40_20/scaffold_graphs/scaffold_F5-F0.pdf} &
    \includegraphics[width=0.17\linewidth]{figs/02_factor_trace/mnist_digit_mlp_40_20/scaffold_graphs/scaffold_F6-F0.pdf} &
    \includegraphics[width=0.17\linewidth]{figs/02_factor_trace/mnist_digit_mlp_40_20/scaffold_graphs/scaffold_F7-F0.pdf} &
    \includegraphics[width=0.17\linewidth]{figs/02_factor_trace/mnist_digit_mlp_40_20/scaffold_graphs/scaffold_F8-F0.pdf} &
    \includegraphics[width=0.17\linewidth]{figs/02_factor_trace/mnist_digit_mlp_40_20/scaffold_graphs/scaffold_F9-F0.pdf} \\
    Factor 5 & Factor 6 & Factor 7 & Factor 8 & Factor 9
  \end{tabular}
  \caption{Scaffold graphs for first 10 factor circuits in the 40$\times$20 MLP. Each graph uses a distinct sparse subset of the 40 L1 neurons, confirming largely non-overlapping feature detectors per digit class.}
  \label{fig:digit-scaffold}
\end{figure}

\newpage
\subsection{Factor Trees and Fingerprints}
\label{sec:fingerprints}

\subsubsection{Active Factor Trees}

Each node in the BFT tree holds a set of NMF img\_factors: per-stimulus loadings that encode how strongly each sample activates the circuit at that position in the hierarchy.
To visualise this, we select the top-$N_s$ stimuli by loading for each tree node and colour every node by their mean loading — producing an \emph{active factor tree} that shows which parts of the circuit hierarchy light up for a given class.

\cref{fig:factor-tree-id} shows the active factor trees for the even/odd MLP (trained on digits $\{0,1,3,4\}$), comparing training and test stimuli of each class.
Training and test sets of the same class produce visually similar activation patterns across all tree nodes, confirming that the discovered circuits generalise within the training distribution and are not overfit to individual samples.
Even and odd circuits activate distinct parts of the tree, visually summarising the class separation already implied by the scaffold graphs.

\begin{figure}[t]
  \centering
  \begin{tabular}{cc}
     \includegraphics[width=0.67\linewidth]{figs/06_stimulus_factor_analysis/tree_output_factors.png}  & \includegraphics[width=0.28\linewidth]{figs/06_stimulus_factor_analysis/similarity_heatmap.png} \\
     (a) Active factor trees & (b) Fingerprint similarity
  \end{tabular}
  
  \caption{(a) Active factor trees for the even/odd MLP, colored by mean img\_factor loading of class-selected stimuli. Left: even circuit. Right: odd ciruit. (b) Pairwise cosine similarity matrix of factor fingerprints for the even/odd MLP test set (stimuli grouped by class). High intra-class similarity and low inter-class similarity confirm that the circuit tree encodes class-relevant structure.}
  \label{fig:factor-tree-id}
\end{figure}

\subsubsection{Factor Fingerprints}

A \emph{factor fingerprint} $\mathbf{f}_s \in \mathbb{R}^{\sum_i K_i}$ is the concatenation of $W_\mathrm{img}[s,:]$ over all BFT tree nodes in breadth-first order, where $K_i$ is the number of NMF components at node $i$.
The fingerprint captures the full hierarchical activation pattern of stimulus $s$ across the entire circuit tree — not just the output-layer factorization.

Pairwise cosine similarity between fingerprints (\cref{fig:fingerprint-similarity}) shows high within-class similarity and low between-class similarity, confirming that the circuit tree provides a class-discriminative stimulus representation.

\newpage
\subsubsection{MDS vs.\ PCA Embeddings}

To compare the geometric structure encoded by factor fingerprints against raw network activations, we embed the test-set stimuli in two ways: (i) MDS on the pairwise cosine distances of factor fingerprints, and (ii) PCA on the concatenated layer-activation vectors.
\cref{fig:mds-pca-comparison} shows this comparison for all three architectures evaluated in notebooks 06–08: the even/odd MLP, the 10-digit MLP, and the CIFAR-10 CNN.

MDS on factor fingerprints consistently yields better-separated class clusters than PCA on raw activations.
PCA mixes classes in activation space because it captures variance irrespective of task structure; the factor fingerprint, by encoding how each stimulus routes through the full circuit hierarchy, clusters stimuli by their computational pathway and thus by class.
This holds across all three architectures and validates that the traced circuits encode task-relevant structure that is not directly visible in the raw activation geometry.

\begin{figure}[t]
  \centering
  \begin{tabular}{c}
       \includegraphics[width=0.9\linewidth]{figs/06_stimulus_factor_analysis/similarity_mds.png} \\
       \includegraphics[width=0.9\linewidth]{figs/06_stimulus_factor_analysis/far_ood_mds.png} \\
       \includegraphics[width=0.9\linewidth]{figs/06_stimulus_factor_analysis/40_20_similarity_mds.png} \\
       \includegraphics[width=0.9\linewidth]{figs/cnn_similarity_mds.png}
  \end{tabular}
  
  \caption{MDS on factor fingerprint cosine distances (left column) vs.\ PCA on raw network activations (right column) for the even/odd MLP (top), 10-digit MLP (middle), and CIFAR-10 CNN (bottom). Points are coloured by class. Factor fingerprint embeddings show consistently cleaner class separation across all architectures.}
  \label{fig:mds-pca-comparison}
\end{figure}

\newpage
\subsection{Validation}
\label{sec:validation}

All validation experiments use the 40$\times$20 digit MLP, run across 5 independent training seeds unless otherwise noted.

\subsubsection{NMF Factor Stability}
\label{sec:stability}

A potential concern is that NMF factorizations are non-unique: different random initializations could yield different components.
We assess stability by running NMF 10 times with different random seeds on the same joint arbor matrix, then computing pairwise cosine similarities after optimal Hungarian matching of components.

\cref{fig:stability} shows per-layer boxplots of matched cosine similarities.
The off-diagonal mean at L1 is $0.938 \pm 0.083$: different seeds converge to the same underlying circuit structure.
Similarity decreases at deeper layers, consistent with smaller arbor matrices and fewer possible components.

\begin{figure}[t]
  \centering
  \includegraphics[width=0.9\linewidth]{figs/03_ablation/fig1b_stability_boxplots.pdf}
  \caption{NMF factor stability across 10 random seeds, using Hungarian matching and cosine similarity at different layers. Off-diagonal mean at L1 is $0.938 \pm 0.083$; factors are highly reproducible despite NMF's non-uniqueness.}
  \label{fig:stability}
\end{figure}

\newpage
\subsubsection{Causal Circuit Ablation}
\label{sec:causal}

Do the traced circuits \emph{cause} the corresponding class decisions, or merely correlate with them?
We perform a targeted ablation: for each class $c$, we zero out all neurons whose traced importance (NMF neural factor loading summed over the circuit) exceeds the $90^\mathrm{th}$ percentile, then measure the change in classification accuracy separately for class $c$ (the \emph{target class}) and all other classes (\emph{bystander classes}).

Results across 5 training seeds (\cref{fig:causal}) show:
\textbf{target class accuracy drop}: $0.848 \pm 0.181$; \textbf{bystander accuracy drop}: $0.505 \pm 0.233$.
A Wilcoxon signed-rank test confirms the difference is significant ($p < 0.0001$), establishing that the traced circuits are \emph{selectively} necessary for their target class.

\begin{figure}[t]
  \centering

  \includegraphics[width=0.9\linewidth]{figs/03_ablation/fig2a_causal_curves.pdf}
  \caption{Causal ablation: Accuracy drop as a function of importance percentile threshold, averaged over classes and seeds. Target classes (solid) drop faster than bystander classes (dashed). Target drop is significantly larger ($p < 0.0001$, Wilcoxon).}
  \label{fig:causal}
\end{figure}

\newpage
\subsubsection{Comparison with Ablation Baselines}
\label{sec:baselines}

We compare the per-neuron importance scores produced by our method against four baselines: weight magnitude, activation-weighted magnitude, Taylor (gradient $\times$ input) attribution, and random.
For each method we sweep the ablation threshold and measure accuracy drop on the target class and on bystander classes separately (\cref{fig:ablation-curves}).

The NMF trace (algo\_top) shows the strongest class-specificity: target-class accuracy drops faster than bystander accuracy, and bystander-class accuracy is preserved better than with magnitude or Taylor baselines.
The least-important ablation (yellow) preserves accuracy across all methods, confirming that the effect is driven by the high-importance neurons.
The AUC advantage of algo\_top over random is significant ($p < 0.0001$, paired $t$-test across seeds).

\begin{figure}[t]
  \centering
  \begin{tabular}{c}
    \includegraphics[width=0.9\linewidth]{figs/03_ablation/per_class_accuracy.pdf} \\
    \includegraphics[width=0.9\linewidth]{figs/03_ablation/bystander_accuracy.pdf}
  \end{tabular}
  \caption{Ablation baseline comparison.
  \textbf{Top}: per-class accuracy drop curves for each method; our trace (red) drops target-class accuracy faster than bystander-class accuracy. Least-important ablations (yellow) preserve accuracy.
  \textbf{Bottom}: bystander-class accuracy for our method and attribution baselines. Ablating the circuit edges identified by our method damages bystander classes less than magnitude or gradient baselines, confirming class-specificity.}
  \label{fig:ablation-curves}
\end{figure}

\newpage
\subsubsection{Comparison with Attribution Baselines}

We compare the pixel-space attribution produced by our method against saliency maps and integrated gradients (\cref{fig:gradient-comparison}).
Our attribution is derived from the L1 neural factor: the factor loadings are reshaped to pixel space to give a per-pixel importance map.
Qualitatively, our maps are sparser and more localized to task-relevant strokes; saliency maps are diffuse and integrated gradients show edge artifacts.
Quantitatively, our method achieves the highest class discriminability score ($0.658$ vs.\ integrated gradients $0.422$, weight magnitude $0.378$, saliency maps near random at $0.103$), meaning its attributions concentrate importance on the target class substantially more than competing methods.

\begin{figure}[t]
  \centering
  \begin{tabular}{c}
    \includegraphics[width=0.9\linewidth]{figs/03_ablation/fig3a_attribution_grid.pdf} \\
    \includegraphics[width=0.9\linewidth]{figs/03_ablation/fig3bc_attribution_metrics.pdf}
  \end{tabular}
  \caption{Gradient methods comparison.
  \textbf{Top}: pixel-space attribution maps for our method (top row), saliency maps (middle), and integrated gradients (bottom), for representative test samples.
  \textbf{Bottom}: class discriminability scores and AUC comparison across methods. NMF trace achieves the highest class-discriminative attribution ($0.658$ vs.\ IG $0.422$, magnitude $0.378$).}
  \label{fig:gradient-comparison}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=0.9\linewidth]{figs/CNN_factors.png}
    \caption{Final-layer NMF factors of the CIFAR-10 CNN. Each factor captures a distinct class-specific response, analogous to the digit-level factors in the MLP experiments.}
    \label{fig:cifar-last-layer}
\end{figure}

\begin{figure}[h]
    \centering
    \begin{tabular}{c}
         \includegraphics[width=0.9\linewidth]{figs/Truck_factors.png}\\
         \includegraphics[width=0.9\linewidth]{figs/Truck_examples.png}
    \end{tabular}
    \caption{Truck circuit. \textbf{Top}: NMF factors reveal two sub-circuits — one for generic trucks and one for fire engines. \textbf{Bottom}: example images from each sub-circuit confirm the split.}
    \label{fig:cifar-truck}
\end{figure}

\begin{figure}[h]
    \centering
    \begin{tabular}{c}
         \includegraphics[width=0.9\linewidth]{figs/Car_factors.png}\\
         \includegraphics[width=0.9\linewidth]{figs/Car_examples.png}
    \end{tabular}
    \caption{Car circuit. \textbf{Top}: factors split the car class into grey and red sub-circuits. \textbf{Bottom}: example images confirm the color-based split.}
    \label{fig:cifar-car}
\end{figure}

\newpage
\subsection{Preliminary: CIFAR-10 CNN}
\label{sec:cifar}

To test generalization beyond MLPs, we extend the backward factor trace to a small CNN trained on CIFAR-10 (10 natural image classes).
Convolutional layers require an adaptation of the joint arbor construction: for a convolutional weight $W \in \R^{C_\mathrm{out} \times C_\mathrm{in} \times k_H \times k_W}$, the arbor at each spatial location would produce a matrix of dimension $N \times (C_\mathrm{out} \cdot C_\mathrm{in} \cdot k_H \cdot k_W \cdot H \cdot W)$, which is intractable.
We address this by \textbf{spatial average pooling}: before computing the arbor, average the spatial input feature maps over the spatial dimensions, collapsing $H \times W$ to a single vector per channel.
This reduces the joint arbor to $N \times (C_\mathrm{out} \cdot C_\mathrm{in} \cdot k_H \cdot k_W)$, which is feasible.
We also apply a confidence pre-filter (top-50 most-confident test samples per class) to reduce $N$ further.

\newpage
The resulting circuits show class-specific spatial activation patterns: the final-layer factors cleanly separate the ten CIFAR-10 classes (\cref{fig:cifar-last-layer}).
Crucially, within each class the method finds meaningful sub-circuits corresponding to within-class variation.
For trucks (\cref{fig:cifar-truck}), two factors emerge corresponding to generic trucks and fire engines.
For cars (\cref{fig:cifar-car}), the split is by color: grey vs.\ red cars.
For horses (\cref{fig:cifar-horse}), the split is by orientation: left-facing vs.\ right-facing animals.
These within-class sub-circuits are not labeled in the training data; they emerge purely from the structure of the joint arbor matrix.



\begin{figure}[h]
    \centering
    \begin{tabular}{c}
         \includegraphics[width=0.9\linewidth]{figs/Horse_factors.png}\\
         \includegraphics[width=0.9\linewidth]{figs/Horse_examples.png}
    \end{tabular}
    \caption{Horse circuit. \textbf{Top}: factors split the horse class by viewing direction — right-facing and left-facing horses. \textbf{Bottom}: example images confirm the orientation-based split.}
    \label{fig:cifar-horse}
\end{figure}

\newpage
\subsection{Preliminary: Vision Transformer}
\label{sec:transformer}

We test the algorithm on a tiny Vision Transformer (TinyViT) trained on MNIST even/odd:
\begin{equation*}
  28\!\times\!28 \;\to\; 16 \text{ patches of } 7\!\times\!7 \;\to\; \text{embed } (49 \to 64) + \text{CLS} + \text{pos.\ embed}
  \;\to\; 2 \times [\text{Attn} + \text{FFN}(64\!\to\!128\!\to\!64)] \;\to\; \text{Linear}(64,2).
\end{equation*}
We apply the backward factor trace to the four FFN linear layers in reverse order, using the \textbf{CLS token activation} at each FFN boundary as the per-sample feature vector.
The CLS token has shape $(N, d)$, which is identical to the MLP layer input format, so \texttt{trace\_single\_layer} applies without modification.

The algorithm cleanly recovers even/odd circuits through the FFN sublayers (\cref{fig:vit-factors}): the final-layer factors separate even and odd stimuli, and tracing back one layer reveals sub-factors corresponding to distinct stroke variations of the digit one.
This demonstrates that the backward trace finds meaningful structure not just at the output but in intermediate FFN layers of a Transformer.

A limitation of the current approach is that it does not trace through the attention sublayers, treating the transformer as a composition of MLPs connected by (unanalyzed) residual streams.
Extending the joint arbor construction to attention weight matrices is left to future work.

\begin{figure}
    \centering
    \begin{tabular}{c}
        \includegraphics[width=0.9\linewidth]{figs/ViT_factors.png} \\
        (a) ViT last layer factors\\
        \includegraphics[width=0.9\linewidth]{figs/ViT_traceback.png} \\
        (b) ViT previous layer factors\\
    \end{tabular}
    
    \caption{Vision Transformer factor trace (MNIST even/odd). \textbf{Top}: NMF factors of the final FFN layer separate even and odd stimuli; the ``ones'' factor is highlighted. \textbf{Bottom}: tracing the ones factor one layer back reveals sub-factors corresponding to different stroke variations of the digit one in the preceding FFN layer.}
    \label{fig:vit-factors}
\end{figure}

\newpage
\section{Discussion}
\label{sec:discussion}

\paragraph{What weight-activation products reveal.}
The central finding is that factorizing weight-activation products rather than activations alone exposes circuit structure that activation-only methods cannot access.
An activation value records whether a neuron fires; the synaptic arbor records \emph{why} it fires, assigning credit to each upstream weight-input pair.
The practical consequence is visible in the attribution comparison (\cref{sec:baselines}): our pixel-space maps concentrate importance on task-relevant strokes (class discriminability $0.658$) while saliency maps scatter it near-uniformly ($0.103$).
The difference is not just quantitative — it reflects a qualitative shift from observing outputs to inspecting the weight structure that produced them.

\paragraph{Excitatory/inhibitory complementarity.}
A consistent pattern across all experiments is that excitatory and inhibitory circuits at the final layer form complementary pairs: the even circuit excites the even output neuron and suppresses the odd one, and vice versa.
This mirrors the known push-pull structure of binary classifiers, but the joint arbor makes it explicit at the level of individual weights rather than inferred from gradient signs.
The inhibitory factors are computed per-layer as diagnostics (\cref{sec:joint-arbor}) but are not propagated backward; they encode which input patterns are actively suppressed rather than promoted, which is a different and complementary form of circuit analysis.

\paragraph{Emergent sub-circuits without labels.}
The CIFAR-10 results (\cref{sec:cifar}) reveal within-class structure that the training labels do not specify: truck vs.\ fire engine, grey vs.\ red car, left- vs.\ right-facing horse.
These splits are not artifacts of arbitrary NMF initialization — the stimulus images retrieved under each factor are visually coherent and the split is consistent across samples.
This suggests that the joint arbor matrix encodes perceptual distinctions that the network uses internally even when the task does not require them, an observation consistent with evidence that neural networks develop richer representations than the label set demands~\citep{zeilerVisualizingUnderstandingConvolutional2013}.

\paragraph{Circuit sparsity and asymmetry.}
The odd circuit in the even/odd MLP (\cref{sec:even-odd}) is strikingly sparse — a single chain through one L2 and one L1 neuron — while the even circuit is more distributed.
This asymmetry suggests the network solves the two sub-tasks with qualitatively different computational strategies, and that sparsity is a meaningful property of the circuit, not an artifact of the factorization.
The scaffold graph representation makes this structural asymmetry legible at a glance in a way that activation-space methods do not support.

\paragraph{Relation to mechanistic interpretability.}
The backward factor trace is complementary to Sparse Autoencoder (SAE) approaches~\citep{cunningham2023sparse} and Transcoders~\citep{lindsey2025transcoders}.
SAEs decompose residual-stream activations into a learned dictionary; Transcoders map MLP computations to input-output functions.
Both require an auxiliary model trained on the network's internal representations.
Our method requires no such auxiliary training: the joint arbor is a deterministic function of existing weights and activations, and NMF is applied to find recurring patterns in that matrix.
The two approaches are not mutually exclusive — SAE features could be used to initialize or interpret the NMF factors, and the backward trace could seed Transcoder-style analysis at identified circuit nodes.

\section{Limitations}
\label{sec:limitations}

\paragraph{Scale.}
All results are on small models: MLPs with at most 40 hidden units, a 3-conv CNN, and a 2-block ViT.
The joint arbor matrix has $N \times (d_\mathrm{out} \cdot d_\mathrm{in})$ entries per layer; for wide layers this can become large ($N=2{,}000$ samples, $d_\mathrm{out}{=}d_\mathrm{in}{=}1{,}024$ gives ${\sim}2\times10^9$ entries), requiring spatial pooling or random projection approximations that discard information.
Whether the method yields interpretable circuits in large-scale models (ResNets, GPT-class LMs) is an open question.

\paragraph{Convolutional approximation.}
The spatial average pooling used for convolutional layers (\cref{appx:conv-arbor}) collapses spatial selectivity: two filters that respond to the same feature at different locations become indistinguishable.
Receptive-field position is therefore not recoverable from the current arbor construction for convolutional layers.

\paragraph{Attention layers.}
The ViT trace operates on FFN sublayers only, treating attention outputs as unanalyzed residual signals.
The weight matrices of multi-head attention — query, key, value, and output projections — could in principle be incorporated into a joint arbor, but the appropriate factorization of a bilinear operation is not immediate and is left to future work.

\paragraph{Excitatory trace only.}
The backward propagation uses the excitatory joint arbor ($\mathbf{J}^+$) to update stimulus weights.
Inhibitory circuits are identified per-layer but their backward signal is not propagated (\cref{sec:joint-arbor}).
A method that traces suppression chains backward — identifying which stimuli are most strongly \emph{inhibited} by a circuit — would provide a more complete picture of the network's computation.

\paragraph{NMF non-uniqueness and rank selection.}
NMF solutions are non-unique; the \texttt{structural\_recon} rank-selection heuristic (fraction drop + reconstruction error floor) is motivated empirically but lacks a theoretical guarantee of recovering the ``correct'' number of circuits.
The stability analysis (\cref{sec:stability}) provides empirical evidence that the discovered factors are robust, but does not rule out alternative decompositions at the same rank that would yield a different circuit interpretation.

\paragraph{Evaluation scope.}
The quantitative validation uses MNIST digit classification, where ground-truth circuit structure is partially known (each digit has a distinct visual template).
Extending the ablation and attribution comparisons to tasks with less legible ground truth — sentiment classification, language modeling, protein structure prediction — would strengthen the generalization claim.

\newpage
\section{Conclusion}

We introduced the Backward Factor Trace, a method for decomposing trained neural network decisions into interpretable circuits via NMF factorization of joint weight-activation arbors.
The method is fully automatic (rank selection, backward propagation, branching), requires no gradient computation or reference input, and yields circuits that are causally specific to their target class, stable across NMF random seeds, and more class-discriminative than gradient-based attribution baselines.
Preliminary results on a CIFAR-10 CNN and a tiny Vision Transformer confirm generalization beyond the MLP setting.
Factor fingerprints — per-stimulus concatenations of NMF loadings across the full BFT circuit tree — provide a class-discriminative stimulus embedding that outperforms PCA on raw network activations, further validating that the traced circuits encode task-relevant structure.



\newpage
\bibliographystyle{plainnat}
\bibliography{bib}

\begin{ack}
Use unnumbered first level headings for the acknowledgments. All acknowledgments
go at the end of the paper before the list of references. Moreover, you are required to declare
funding (financial activities supporting the submitted work) and competing interests (related financial activities outside the submitted work).
More information about this disclosure can be found at: \url{https://neurips.cc/Conferences/2026/PaperInformation/FundingDisclosure}.


Do {\bf not} include this section in the anonymized submission, only in the final paper. You can use the \texttt{ack} environment provided in the style file to automatically hide this section in the anonymized submission.
\end{ack}



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\appendix


\newpage
%\input{checklist.tex}

\section{Experimental Details}
\label{appx:experiments}

\subsection{Datasets}

\textbf{MNIST.}
Even/odd experiments restrict to digits $\{0,1,3,4\}$ and relabel them by parity,
giving a binary task on ${\sim}28{,}000$ training images.
Digit-classification experiments use the full $60{,}000$-image training set.
Both use the standard $10{,}000$-image test split.
Analysis activations are collected from correctly-classified test samples only,
with up to 1{,}000 samples per class for the main trace and 200 per class for
the validation experiments.

\textbf{CIFAR-10.}
$50{,}000$/$10{,}000$ train/test images across 10 natural-image classes.
For the trace we retain the top-60 most-confident test samples per class
(selected by max-softmax probability) to keep the joint arbor tractable.

\subsection{Model Architectures}

\textbf{SimpleMLP.}
Configurable fully-connected network: \texttt{Flatten} $\to$ $L$ blocks of
$[\texttt{Linear}, \texttt{Sigmoid}]$ $\to$ $\texttt{Linear} \to \texttt{Softmax}$.
Two configurations are used:

\begin{itemize}
    \item \emph{Even/Odd MLP}: $784 \!\to\! 8 \!\to\! 4 \!\to\! 2$
    \item \emph{Digit MLP}: $784 \!\to\! 40 \!\to\! 20 \!\to\! 10$
\end{itemize}

\textbf{SmallCNN (CIFAR-10).}
Three convolutional blocks (Conv$3\!\times\!3$ + BatchNorm + MaxPool,
channels $3\!\to\!16\!\to\!32\!\to\!64$) followed by $4\!\times\!4$ adaptive
average pooling and two FC layers ($1024\!\to\!128\!\to\!10$).

\textbf{TinyViT.}
$28\!\times\!28$ images are split into 16 non-overlapping $7\!\times\!7$ patches,
linearly embedded to dimension 64, prepended with a learnable CLS token and
positional embeddings, and processed by 2 Transformer blocks
(LayerNorm $\to$ MHA 2-head $\to$ LayerNorm $\to$ FFN $64\!\to\!128\!\to\!64$,
GELU), followed by a linear head to 2 classes.
The trace is applied to the CLS token at each of the four FFN linear-layer
boundaries (B0-FFN1, B0-FFN2, B1-FFN1, B1-FFN2).

\subsection{Training and Hyperparameters}
\label{appx:hyperparams}

\cref{tab:hyperparams} lists all hyperparameters.
MLP models are trained with Adam and cross-entropy loss until convergence
(typically 20 epochs).
The CIFAR-10 CNN is trained with SGD and a cosine annealing schedule for 40 epochs;
train augmentation uses random crop with 4-pixel padding and random horizontal flip,
with per-channel normalization to mean $(0.491, 0.482, 0.447)$ and
std $(0.247, 0.244, 0.262)$.
TinyViT is trained with Adam and NLL loss for 12 epochs.

\begin{table}[h]
\centering
\caption{Hyperparameter summary. BFT reconstruction threshold $\varepsilon_r$ is the
relative Frobenius error floor for rank selection; stimulus threshold $\tau$ zeros
the lowest-$\tau$ fraction of samples by stimulus weight.}
\label{tab:hyperparams}
\begin{tabular}{@{}lll@{}}
\toprule
\textbf{Component} & \textbf{Hyperparameter} & \textbf{Value} \\
\midrule
MLP training   & Optimizer / lr / batch      & Adam / $10^{-3}$ / 32 \\
CNN training   & Optimizer / lr / momentum   & SGD / $0.1$ / $0.9$ \\
               & Weight decay / epochs       & $5\times10^{-4}$ / 40 \\
ViT training   & Optimizer / lr / epochs     & Adam / $10^{-3}$ / 12 \\
\midrule
BFT (MLP) & $k_\mathrm{max}$ [L1, L2, out] & $[20, 15, 15]$ \\
               & Rank-selection method        & \texttt{structural\_recon} \\
               & Recon.\ threshold $\varepsilon_r$ & $0.20$ \\
               & Stimulus threshold $\tau$    & $0.70$ (viz) / $0.10$ (ablation) \\
               & Branches $B$ (even/odd)      & $[10, 2, 2]$ \\
               & Branches $B$ (digit)         & $[1, 1, 10]$ \\
BFT (ViT) & $k_\mathrm{max}$ / $\varepsilon_r$ & $15$ / $0.20$ \\
\midrule
NMF solver     & Library / init               & scikit-learn / \texttt{nndsvda} \\
               & Max iterations               & $20{,}000$ \\
\midrule
Validation     & Training seeds               & 5 (seeds 0--4) \\
               & NMF stability seeds          & 10 \\
               & Samples per class            & 200 (ablation / stability) \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Convolutional Arbor Approximation}
\label{appx:conv-arbor}

For a convolutional layer with weight
$W \in \R^{C_\mathrm{out} \times C_\mathrm{in} \times k_H \times k_W}$
and spatial feature maps of size $H\!\times\!W$, the full per-location joint arbor
would have $C_\mathrm{out} \cdot C_\mathrm{in} \cdot k_H \cdot k_W \cdot H \cdot W$ columns,
which is intractable.
We collapse the spatial dimension by averaging the input feature map over $H\!\times\!W$
before forming the arbor, reducing the column count to $C_\mathrm{out} \cdot C_\mathrm{in} \cdot k_H \cdot k_W$.
For the SmallCNN this yields arbor widths of at most $64 \cdot 32 \cdot 3 \cdot 3 = 18{,}432$;
arbors wider than 1{,}000 features are further reduced by random projection
before NMF.
This approximation discards spatial selectivity within the receptive field but
preserves channel-level co-activation patterns across the filter bank.


\end{document}
