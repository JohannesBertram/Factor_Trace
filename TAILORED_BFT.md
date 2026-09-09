# Tailored stimulus sets for BFT ("dataset probes")

Research branch `tailored-bft`. Implementation: `src/tailored.py` + `notebooks/06_tailored_imagenet.ipynb`. Status: implemented, untested on real ImageNet data (needs `data/val`, i.e. a cluster run).

## The idea

Standard BFT factorizes each layer's joint arbor matrix over a broad, class-balanced stimulus set. In activation-space interpretability terms this is the SAE end of the spectrum: an unsupervised dictionary fit to the full data distribution, whose capacity goes to the dominant global structure — rare or fine-grained concepts get absorbed into coarse factors. The probe end of the spectrum (linear probes, TCAV) is supervised: you curate a dataset embodying one hypothesis and derive a representation for exactly that.

Tailored BFT moves the method toward the probe end **without changing the method**. Because the rows of the joint arbor matrix are stimuli, the stimulus distribution is a free parameter of the factorization: enriching the set with a hypothesis (all bear species, one dog breed group, "striped things") reallocates NMF's reconstruction budget onto the circuitry that processes those stimuli. Curating the stimulus set is pointing the microscope — sub-structure the broad trace blurs into a single factor becomes resolvable. The trace stays fully unsupervised; the only supervision is the choice of stimuli, exactly as with a probe dataset (the closest published analogue is TCAV's curated concept sets, applied here to a weight-space method).

## Why this is the right next experiment for this repo

The error-consistency pilot (research/error_consistency_bft, 2026-07) found that BFT similarity predicts error consistency but is **redundant with raw activations except when the error-relevant sub-structure is not the training target** (even/odd per-digit: BFT 0.26 > act 0.13; validation gate: BFT recovers planted sub-structure at ARI 0.78 where activations get 0.03). Tailored stimulus sets are the tool that deliberately operates in that regime: they interrogate structure below or across the categories any single broad trace is organized around.

One honesty note that shapes the hypothesis taxonomy below: for SqueezeNet the 1000 fine ImageNet classes **are** training targets, so for pure class zoom-ins (bears → species) the penultimate activations will be a strong baseline — the model was trained to separate species. The unique BFT contribution there is mechanistic (which factors, which weights, causal handles), not representational separation. The regime where tailored BFT should also **win on separation** is attribute hypotheses orthogonal to the label space (stripes, habitat, pose) — the true non-target regime from the pilot.

## Hypothesis taxonomy

- **H1 — class zoom** (`bears`, `dogs`): stimulus set = one super-category, expected sub-structure = species/breed groups. Prediction: tailored factors split by sub-class where the broad trace had one category factor; value = sub-circuit identification + causal selectivity, with activations expected to remain a strong separation baseline.
- **H2 — cross-category attribute** (`striped`: zebra/tiger/tiger-cat vs matched plain counterparts sorrel/lion/Egyptian-cat): structure orthogonal to the label space. Prediction: a mid-layer factor loading on striped stimuli across categories — the case where the tailored trace can beat activation baselines outright, per the pilot.
- **H3 — contrastive sets** (`cats` domestic vs big, `canids` dog/wolf/fox, `birds` song/water): hypothesis + control in one set. Readout: which factors are shared (generic texture/shape machinery) vs group-specific; does the trace split by taxonomy or appearance?
- **H0 — null control** (`random_control`: arbitrary classes in arbitrary groups): calibrates how much "sub-structure" NMF invents on sets with no shared circuitry. Every H1–H3 result is read relative to this.

## Research questions

- **RQ1 Resolution.** Does the tailored trace resolve sub-structure the broad trace misses? Metrics (§5 of nb06): fine-class silhouette / 5-NN / k-means-ARI of tailored fingerprints vs (a) the same stimuli NNLS-projected onto the fixed broad nb05 factors, (b) pooled penultimate activations, (c) all-layer pooled activations. Secondary signal: held-out rank re-derivation (§3b) asks for more rank on tailored arbors than the broad profile used.
- **RQ2 Zoom vs rotation.** Is the tailored factorization a *refinement* of the broad one (each tailored factor descends from one broad factor — the category factor splitting) or a *rotation* (new basis)? Metric (§6): connection-factor cosine matching + range-rescaled top-margin (`refinement_score`), calibrated against `random_control`. Refinement supports the "zoom" interpretation; rotation would mean the stimulus distribution changes the basis itself — worth knowing either way.
- **RQ3 Dose–response.** How much enrichment does sub-structure need, and is emergence smooth or abrupt? §7 sweeps the hypothesis fraction ρ at fixed total N and fixed rank profile (both confounds controlled), reading fine-class separability of the hypothesis stimuli at each ρ. A sharp threshold would be the most interesting outcome (factor capacity competition); the ρ at which structure saturates is practical guidance for how to build probe sets.
- **RQ4 Causality.** Is the tailored sub-circuit *the* circuit for its group? §8 prunes top-importance weights of the group-selective tailored circuit and measures damage selectivity (target-group drop minus mean off-target drop) against (a) matched random ablation and (b) the broad tree's circuit selected for the same group via NNLS projection. Prediction: tailored > broad > random selectivity at matched fraction.
- **RQ5 Attribute discovery.** Does `striped` surface a factor unifying stripes across categories, with loading split striped/plain rather than by category? This is the purest probe-style use: dataset-as-hypothesis-test. Follow-ups if it works: texture vs shape sets (Geirhos-style), backgrounds/habitat, canonical pose.

## Validity threats and controls

- **NMF invents structure.** Controls: H0 null hypothesis; NMF stability across seeds (`compute_nmf_stability`, run on any headline hypothesis before believing it); held-out rank criterion rather than fixed generous ranks.
- **Sample-size confound.** Enriched sets must not simply have more images of the hypothesis: RQ3's mixtures fix total N; RQ1 comparisons all score the identical stimulus set.
- **Correctness filter.** `strict` (1000-way top-1 correct) biases toward easy exemplars and shrinks fine-grained sets (SqueezeNet is ~58% top-1, worse on breeds); `group` mode (predicted *some* hypothesis class) keeps stimuli where the model ran the right circuitry but missed the species — arguably the right filter for sub-structure questions. Run headline results under both.
- **Cosine floor.** Non-negative factor vectors have high baseline cosine; `refinement_score` rescales the top-vs-second margin by the row's cosine range, and all correspondence numbers are read against `random_control`.
- **Thin data.** ImageNet val has ~50 images/class; after `strict` filtering a fine class can drop to ~20. Acceptable for first tests; if factors look promising but noisy, move stimulus collection to the train split (guard: the model saw those images — check conclusions hold on val).

## Implementation map (this branch)

- `src/tailored.py`: `find_imagenet_classes` (name → id lookup from torchvision metadata, no download), `collect_hypothesis_data` (hypothesis dict → balanced spine layer dicts with fine labels and `strict`/`group`/`none` correctness modes), `mixture_indices` + `subsample_data` (fixed-N dose–response mixtures), `factor_label_profile` (per-factor mass over labels, purity/entropy), `substructure_scores` (silhouette/kNN/ARI, cosine metric), `match_factors` + `refinement_score` (cross-tree factor correspondence in shared arbor space). All exported via `src/__init__.py`.
- `notebooks/06_tailored_imagenet.ipynb`: self-contained, nb05 conventions (same spine filter, val loader, caching). Hypothesis picked via `TAILORED_HYP` env var (default `bears`); registry in §1 with verified class ids. Env gates: `TAILORED_RANKS=1` (§3b rank re-derivation), `TAILORED_DOSE=1` (§7), `TAILORED_PRUNE=1` (§8), `TAILORED_CORRECT=strict|group|none`. §5/§6/§8 auto-load the newest cached nb05 broad tree (`data/cache/nb05_circuit__*.pkl`) and degrade gracefully without it.
- Adding a hypothesis = adding one dict to `HYPOTHESES` (`groups`, `n_per_class`, `k_root`, `prune_group`); `find_imagenet_classes('husky')` for ids.

## Roadmap

1. **Now (cluster):** run nb05 (to populate the broad-tree cache), then nb06 with `bears`, `dogs`, `striped`, `random_control`; RQ1/RQ2 read directly off §5/§6. Then §7/§8 for the most promising hypothesis.
2. **Next:** systematic H1 sweep over many super-categories (WordNet hierarchy can generate zoom hypotheses automatically); train-split collection for statistical weight; NMF-stability pass on headline factors; `group`-mode vs `strict`-mode comparison.
3. **Later:** attribute library (texture/shape/background/pose probe sets); connect back to the error-consistency thread — tailored traces on ImageNet are exactly its proposed "CNN/ImageNet super-category" decisive next step; if RQ2 shows clean refinement and RQ4 shows selectivity, this becomes a standalone methods contribution ("dataset probes for weight interpretability": SAE→probe spectrum, TCAV positioning) rather than a paper appendix.
