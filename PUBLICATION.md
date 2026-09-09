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
| 1. fig2e even/odd pruning, 5 seeds | **yes** (main-figure panel) | **still open (2026-09-09)**: the per-seed result files turned out to be five identical seed-0 copies (driver bug) — rerun the §8 seed grid with real per-seed checkpoints; see the cluster-run section below |
| 2. figN ImageNet pruning | **no** — paper's limitations already scope ImageNet out of pruning; nothing else depends on it | **resolved (2026-09-09)**: the downsized nb05 §8 run came back null (p_tvb=0.945, 8 categories) — keep excluded with one honest sentence |
| 3. paired CIFAR/ViT `act` baseline | **yes** (fig4 + paired stats) | **resolved by the refactor** — comes for free with the rerun below |
| 4. CIFAR causal-recon panel (figP-a "n/a") | **no** — paper methods scope recon to the two MLPs | optional restore: trace CIFAR in primary mode with a pre-filtered 2000-image loader + `validate=True` |
| 5. decide CIFAR/ViT fingerprint HPs | — | **superseded** by the single-tree decision |

New points this round surfaced:

- **CIFAR model seeds**: the paper claims five trained CIFAR seeds (appendix); locally only seed-0 exists. Verify seeds 1–4 exist on the cluster or train them (`scripts/train_extra_seeds.sh cnn`) — needed for the pruning seed grid and the seed-robustness wording; otherwise weaken the claim to seed-0 + stimulus resampling.
- **Pruning observation counts**: fresh digit/CIFAR pruning is 1 seed × 10 classes; the paper's stats text assumes seed×class grids. Either extend `_reps` in §8 with more checkpoints on the cluster, or fix the text to what was run.
- **fig2e x-axis**: `pruning_panel` still draws xticks at 0/25/50% but the grid now caps at 20% — small `src/paper_figures.py` tweak when re-rendering.
- **ImageNet `act_out`**: not measurable without the val set locally; comes automatically from §7/§9 in the rerun (conv root → no logits control; note this in the paper as n/a).

## Publication refactor (2026-09-08, done locally — this repo state)

The repo was cut down to the publishable core ahead of the cluster rerun; full plan/rationale in the session plan, headline changes:

- **Deleted**: notebooks 08–16, `executed_09_*`, the six `fig0N` notebooks (`scripts/render_figures.py` + `src/paper_figures.py` is the only figure path now); obsolete scripts (`add_activation_baselines`, `last_layer_separability`, the nb09/nb10 cluster drivers); side-project src modules; figdata bundles no figure loads (nb11/12/15/16, error_consistency); tracked `logs/09_validation` PDFs; ~2k lines of dead src code. `pruning.py`'s aggregation merged into `bundles.pruning_bundle` (single stats path).
- **Bundle shrink (export side, before the rerun)**: shared uint8 `stimulus_pool` replaces per-node `top_images` (`paper_figures.top_stims` gathers, backward-compatible); `wavg` uint8; conv `conn` marginals-only via lowered `max_matrix`; nb03/nb05 drop scaffold edges + slim spatial maps; nb04 gains `stim_idx`; every circuit bundle exports full-length `stim_labels` (fixes the CIFAR purity-CI IndexError — fig6d/figN error bars change for the better). Projected sizes: nb02 ~3.5 / nb03 ~10 / nb04 ~4 / nb05 ~12 MB → the whole figdata dir fits GitHub without LFS.
- **§8 rework**: cache key now includes seeds/fractions/eval-size (extending `_reps` recomputes instead of stale-hitting); **ImageNet pruning downsized and moved into nb05 §8** — seeded ≤50 images/category eval subsample (~400), 4-point grid to 20%, `n_random=3`, category-mapped accuracy via `pred_transform`. Cancel the old long-running nb14-style cluster job; its replacement runs inside nb05.
- Verified locally: nb01 end-to-end (all four bundle writes, caches hit, pruning floor rejects the smoke run), 12/12 registry figures render from mixed old/new bundles, `compute_checklist_stats.py` completes (stale-bundle sections skip cleanly until the rerun).
- **docs/ website is frozen**: `docs/build_data.py` is compatible with the new bundle format but must NOT be run until the updated circuits are ready (user supplies them later).
- After the rerun + figure verification: create the fresh squashed public history (orphan branch / new repo) — the old 981 MB `.git` contains a >100 MB blob and cannot push to GitHub.

## Cluster run 2026-09-09 — verification, results, remaining gaps

The rerun landed (`3b9f274 new res`); bundles verified 2026-09-09. Status per model:

- **Complete (nb01, nb02, nb04, nb05)**: all bundles present and new-format (uint8 `images` pool, `stim_labels`, no per-node `top_images`); fingerprint bundles carry the aligned `act` baseline (`aligned=1`); validation JSONs in `logs/results/` are fresh (Sep 9). Sizes: nb01 1.2 / nb02 2.8 / nb04 8.8 / nb05 19.4 MB circuits — everything comfortably under GitHub limits, no LFS needed.
- **nb03 (CIFAR) died after §3**: `nb03_circuits` arrived (15.5 MB, new format), but `nb03_fingerprints`, `nb09_cnn_cifar_validation`, and the §8 pruning bundle are missing, and `logs/results/nb09_cnn_cifar.json` is still the stale Sep 7 one. **Action: rerun nb03 §4–§9 on the cluster** (the §2 tree cache hits, so this is cheap). This blocks fig4, figP, figfp_ood, figfp_structure and the CIFAR entries of the checklist (silhouette expected ≈0.665 per the top-2 experiment).
- **Pruning bundles**: ImageNet's downsized §8 ran and passed the floor — 8 category-observations, 4-point grid, **p_tvb=0.945: the honest null** — keep ImageNet excluded from the pruning claims with one sentence, as already planned. Digit MLP: 10 obs, p=0.002, but **1 seed × 10 classes** — the paper's stats text must say exactly that (or extend `_reps` with seed 1–4 checkpoints in a future pass). CIFAR: bundle restored from the pre-cleanup state (refactor-era 1 seed × 10 classes, p=0.002; valid — pruning runs on the circuit tree, which the single-tree fingerprint change does not touch).
- **Even/odd 5-seed pruning is NOT recovered**: the five per-seed result files (`data/results/nb13_pruning_mlp_even_odd{0..4}.json`) are **byte-identical copies of the seed-0 run** — the driver never switched checkpoints, so no 5-seed data exists. The pre-cleanup 10-observation bundle was restored so fig2e renders, but it is old-schema and its seed provenance cannot be verified from the bundle. **Action: rerun the even/odd §8 seed grid with `_reps` actually loading the seed-0–4 checkpoints, and check the resulting `obs_seed` really spans five seeds.**
- **Fresh headline numbers** (checklist, native-dim silhouettes): even/odd parity 0.925 / digit-structure **0.383** (as predicted); digit MLP **0.665** (up from 0.554, beats dim-matched act 0.221 by +0.444); **ImageNet 0.432** [0.410, 0.467] — the feared drop to ~0.30 did not materialize (old two-tree was 0.476; fp beats matched act 0.206 with P(fp>act)=1.000); ViT 0.208 parity / 0.137 digit (still the feasibility case). Weight-term control: 4/4 available models arbor-NMF > activation-NMF. CIFAR purity CI now computes with real error bars from `stim_labels` (output-layer 0.507, the §1.3 bug fix working); NMF stability min 0.816 (one layer below the 0.85 gate — same as before, mention in text).
- **Figures**: 8/12 regenerated from the new bundles (fig2, figA, figB, fig6, figE, fig8, figN, figG); fig4/figP/figfp_ood/figfp_structure wait on nb03. Small code changes in this pass: fig6 takes purity labels from the circuit bundle's own `stim_labels` (no more `nb03_fingerprints` dependency), `render_figures.py` and `compute_checklist_stats.py` skip missing bundles gracefully instead of aborting.

## Run plan

**Cluster (one pass):**
1. Pull this repo state. Run notebooks 01–05 (`nbconvert --execute`, same commands as before). Each writes all four of its bundles into `figures/figdata/`. For the pruning seed grids, extend `_reps` in §8 with the extra checkpoints (nb01: seeds 0–4; nb02/nb03 if checkpoints exist). Circuit-tree caches still hit; §7/§9 recompute at top2; §8 recomputes under the new key. nb05 §8 now runs the downsized ImageNet pruning itself — cancel the old in-flight nb14 job.
2. Copy `figures/figdata/` here (plus `logs/results/`, `data/results/` for provenance if convenient). Check no bundle exceeds ~10 MB (`find figures/figdata -size +10M`).

**Local (after figdata arrives):**
4. `python scripts/render_figures.py` — then eyeball fig4 and figfp_ood/figfp_structure closely: the fingerprint dimensionality changed a lot (e.g. even/odd 13→5, ImageNet 236→~105), so panel layouts may need adjusting. Do NOT rebuild `docs/` — the website waits for the user-supplied updated circuits.
5. `python scripts/compute_checklist_stats.py` → re-paste every number/CI into `paper.md`. The previously-skipping sections (CIFAR fp/act paired, CIFAR purity CI) must now compute; the ImageNet pruning stats text must state the downsized protocol (≤50 imgs/category, 4-point grid, n_random=3).
6. `paper.md` text pass: replace the two-tree description (§2.4 "Two trees" paragraph, Table 3's fingerprint-tree block, Appendix rank-sweep last paragraph, validation-suite mentions) with the top-2-slice definition; update the separability discussion per the comparison rules above (logits control, PCA-matched penultimate); update stale captions (node counts figE 131 / figN 273 / figG 51; fig2e axis; the `%` comment block above the pruning stats in appx marks the numbers to re-paste); rescope the headline: CIFAR strengthens, ImageNet fingerprint number shrinks but stays a clear win, even/odd fine story survives at 0.38, ViT stays a feasibility demo.
7. One caveat to write honestly: on the tiny even/odd MLP, PCA-compressing the **full activation concatenation** to the fingerprint's 5 dims scores 0.571 on digit labels — above the 5-dim fingerprint's 0.383 (nb01 §7 verified locally). At very small matched dims, PCA concentration favors wide activation stacks; the fingerprint's wins are the conv models and the depth/same-layer-matched comparisons. Frame the claim accordingly rather than as a universal "beats activations".

## Optional / later (not blockers)

- CIFAR primary-mode trace for the causal-recon panel (drops the figP-a "n/a").
- λ-weighted whole-tree fingerprint as a one-tree alternative that might keep ImageNet's 0.476 — untested; only worth it if the ImageNet regression bothers a reviewer.
- ImageNet stimulus scale-up (S=544 is val-limited; relax `only_correct` or add fine classes per category).
- nb15 ViT sweep was cap-bound at k_cap=10 in two layers; re-run with 16 if the ViT ever becomes more than a feasibility demo.
