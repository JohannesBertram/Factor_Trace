# PUBLICATION.md — open points and run plan

Single source of truth for what remains before the paper is submittable (2026-09-07). Supersedes the notes now in `archive/` (REGEN_STATUS, EXPERIMENTS_TO_RUN, REBUTTAL_PLAN, PUBLICATION_SETTINGS, LAST_LAYER_FP_FINDINGS, …).

## Decision: single tree, fingerprint = top-2 slice

The two-tree design is dropped. Each model fits **one circuit tree** (nb15 held-out HPs, unchanged), and the fingerprint is the tree's **top two levels** — the output-layer factors plus their immediate sub-circuits (`truncate_tree(tree, depth=2)` in `src/fingerprint_utils.py`). This is implemented in notebooks 01–05 already; the fingerprint/OOD/separability results just need one cluster rerun to regenerate.

Evidence (`scripts/last_layer_separability.py`, full detail in `archive/LAST_LAYER_FP_FINDINGS.md`; silhouette, seed-0):

| model | top-2 fp (new) | output-only | two-tree fp (old) | PCA-matched penult | logits (`act_out`) |
|---|---|---|---|---|---|
| cnn_cifar | **0.665** | 0.675 | 0.362 (tied acts) | 0.589 | 0.765 |
| mlp_digit | **0.665** | 0.729 | 0.559 | 0.856 | 0.862 |
| mlp_even_odd (fine) | **0.383** | 0.029 | 0.559 | ~0.08 | 0.029 |
| vit_mnist (task/fine) | 0.215 / 0.124 | 0.272 / 0.091 | 0.324 / 0.118 | 0.268 / 0.116 | 0.520 / 0.065 |
| imagenet_cnn | ~0.297 (nb16) | 0.297 | 0.476 | 0.249 | not measured |

What this buys and costs: **CIFAR is repaired** (0.362→0.67, from a tie with matched activations to a clear win — the old regression was dilution by deep low-purity factors), the story simplifies to one tree, and the fingerprint becomes cheap (top-2 needs no deep trace). **Accepted regressions**: ImageNet drops 0.476→~0.30 (still ≫ activations 0.12–0.25), even/odd digit structure drops 0.559→0.38 (still beats depth-matched activations 0.32), ViT stays the weak case (present as feasibility, as the paper already does).

Comparison rules for the paper (from the same experiment): the primary control is the **PCA-dim-matched penultimate**; report the **logits** (`act_out`) too and say why they are excluded as a "representation" (they are the decision itself — trained on exactly the evaluation labels, they beat every code at the output layer); depth-matched concatenations (`act_last2`, `act_full`) are the fair opponents for multi-layer codes; kNN saturates and does not discriminate — the claim is a silhouette/geometry claim. These baselines are now computed by §7/§9 automatically (`src/separability.py`, `src/validation.py`).

## What the refactor changed (2026-09-07, this repo state)

- Notebooks 01–05 are **model-wise complete and self-contained**: one run writes `nb0N_circuits`, `nb0N_fingerprints` (now including the aligned `act` baseline in-notebook — `scripts/add_activation_baselines.py` is obsolete), `nb09_<exp>_validation`, and the pruning bundle, all directly into `figures/figdata/` (`src/bundles.py`). **Only `figures/figdata/` needs to move off the cluster.**
- §4/§5 fingerprint/OOD sections run on the top-2 slice of the circuit tree; no second tree is fitted; `K_MAX_FP`/`N_BRANCHES_FP` are gone from the configs.
- §7/§9 result caches are auto-busted (`fp='top2'` in the cache params), so a rerun recomputes at the new definition; circuit-tree caches still hit.
- `scripts/build_validation_bundles.py` / `build_pruning_bundle.py` remain as thin rebuilders from stored results JSONs.
- The **weight-term control was fixed** (`src/separability.weight_term_control`): it now matches node for node — every tree node, same rank, same stimulus weighting on the activation side — instead of one ungated node per layer. On nb01 the asymmetric version produced an artifactual 0.05-vs-0.29 loss; the matched version gives arbor 0.383 vs activation-NMF 0.360. figP-e's numbers will move accordingly in the rerun; the claim direction should be re-checked per model.
- nb01 verified end-to-end locally (all four bundles written; the pruning-bundle floor correctly rejected the 2-observation local smoke run).

## Old open points — necessary? rerun?

| open point (was in REGEN_STATUS) | necessary? | status / action |
|---|---|---|
| 1. fig2e even/odd pruning, 5 seeds | **yes** (main-figure panel) | you ran it — when the results JSON lands, `python scripts/build_pruning_bundle.py mlp_even_odd` (or it builds automatically if §8 ran in the refactored notebook) |
| 2. figN ImageNet pruning | **no** — paper's limitations already scope ImageNet out of pruning; nothing else depends on it | running on cluster; if the per-layer-sparsity result is clean, add the panel + a sentence; if null, keep excluded (no silent drop — one honest sentence) |
| 3. paired CIFAR/ViT `act` baseline | **yes** (fig4 + paired stats) | **resolved by the refactor** — comes for free with the rerun below |
| 4. CIFAR causal-recon panel (figP-a "n/a") | **no** — paper methods scope recon to the two MLPs | optional restore: trace CIFAR in primary mode with a pre-filtered 2000-image loader + `validate=True` |
| 5. decide CIFAR/ViT fingerprint HPs | — | **superseded** by the single-tree decision |

New points this round surfaced:

- **CIFAR model seeds**: the paper claims five trained CIFAR seeds (appendix); locally only seed-0 exists. Verify seeds 1–4 exist on the cluster or train them (`scripts/train_extra_seeds.sh cnn`) — needed for the pruning seed grid and the seed-robustness wording; otherwise weaken the claim to seed-0 + stimulus resampling.
- **Pruning observation counts**: fresh digit/CIFAR pruning is 1 seed × 10 classes; the paper's stats text assumes seed×class grids. Either extend `_reps` in §8 with more checkpoints on the cluster, or fix the text to what was run.
- **fig2e x-axis**: `pruning_panel` still draws xticks at 0/25/50% but the grid now caps at 20% — small `src/paper_figures.py` tweak when re-rendering.
- **ImageNet `act_out`**: not measurable without the val set locally; comes automatically from §7/§9 in the rerun (conv root → no logits control; note this in the paper as n/a).

## Run plan

**Cluster (one pass):**
1. Pull this repo state. Run notebooks 01–05 (`nbconvert --execute`, same commands as before). Each writes all four of its bundles into `figures/figdata/`. For the pruning seed grids, extend `_reps` in §8 with the extra checkpoints (nb01: seeds 0–4; nb02/nb03 if checkpoints exist).
2. Let the already-running ImageNet pruning finish; it writes `nb14_pruning_imagenet_cnn.json` → bundle via `build_pruning_bundle.py imagenet_cnn` (or rerun nb05 §8 under the refactor).
3. Copy `figures/figdata/` here (plus `logs/results/`, `data/results/` for provenance if convenient).

**Local (after figdata arrives):**
4. `python scripts/render_figures.py` — then eyeball fig4 and figfp_ood/figfp_structure closely: the fingerprint dimensionality changed a lot (e.g. even/odd 13→5, ImageNet 236→~105), so panel layouts may need adjusting.
5. `python scripts/compute_checklist_stats.py` → re-paste every number/CI into `paper.md`.
6. `paper.md` text pass: replace the two-tree description (§2.4 "Two trees" paragraph, Table 3's fingerprint-tree block, Appendix rank-sweep last paragraph, validation-suite mentions) with the top-2-slice definition; update the separability discussion per the comparison rules above (logits control, PCA-matched penultimate); update stale captions (node counts figE 131 / figN 273 / figG 51; fig2e axis; the `%` comment block above the pruning stats in appx marks the numbers to re-paste); rescope the headline: CIFAR strengthens, ImageNet fingerprint number shrinks but stays a clear win, even/odd fine story survives at 0.38, ViT stays a feasibility demo.
7. One caveat to write honestly: on the tiny even/odd MLP, PCA-compressing the **full activation concatenation** to the fingerprint's 5 dims scores 0.571 on digit labels — above the 5-dim fingerprint's 0.383 (nb01 §7 verified locally). At very small matched dims, PCA concentration favors wide activation stacks; the fingerprint's wins are the conv models and the depth/same-layer-matched comparisons. Frame the claim accordingly rather than as a universal "beats activations".

## Optional / later (not blockers)

- CIFAR primary-mode trace for the causal-recon panel (drops the figP-a "n/a").
- λ-weighted whole-tree fingerprint as a one-tree alternative that might keep ImageNet's 0.476 — untested; only worth it if the ImageNet regression bothers a reviewer.
- ImageNet stimulus scale-up (S=544 is val-limited; relax `only_correct` or add fine classes per category).
- nb15 ViT sweep was cap-bound at k_cap=10 in two layers; re-run with 16 if the ViT ever becomes more than a feasibility demo.
