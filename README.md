# Foldable but not dockable

**An over-squashing account of AlphaFold's protein–protein interface failures on PINDER.**
Graph Representation Learning mini-project, TU Wien.

Murat Kolic — matrikelnummer 12551216.

AlphaFold folds each chain confidently but a large fraction of its *complex* predictions are wrong —
the well-documented "great monomers, weak complexes" regime. Why should placing two well-folded chains
relative to one another be so much harder than folding each? I reframe this as a property of
message-passing GNNs: **over-squashing**, the exponential compression of information forced through a
graph bottleneck. Modelled as a graph, a folded chain is internally dense, so intra-chain reasoning has
no bottleneck; the interface between two chains is a thin neck of a few contacts through which all
cross-chain signal must squeeze — exactly the topology in which message passing over-squashes.

I study AlphaFold's interface **sub-task** — predicting which residues sit at the inter-chain interface
— with my own GNN, on **PINDER's leakage-free** split, using AlphaFold's **predicted monomers (+pLDDT)**
as input and the bound complex only for labels. There is **no folding inference anywhere**, which
sidesteps needing to run docking and keeps the study reproducible on a single GPU.

The report is in [`documentation/report.pdf`](documentation/report.pdf).

---

## Headline result: over-squashing is contingent on feature richness

The textbook prediction is that rewiring relieves a bottleneck, so it should help. That turns out to
hold only in a specific regime. Holding the GNN architecture fixed and varying **only** the per-residue
node features along a richness ladder (each regime self-consistent — its cross-chain kNN layer rebuilt
from its own features), the benefit of cross-chain rewiring **flips sign**:

| node features | dim | none (AUPRC) | knn (AUPRC) | **ΔAUPRC (knn − none)** |
|---|---|---|---|---|
| baseline (degree + pLDDT) | 3 | 0.209 | 0.295 | **+0.086** |
| one-hot amino acid | 24 | 0.196 | 0.232 | **+0.036** |
| ESM2-32 | 35 | 0.201 | 0.202 | **+0.001** |
| ESM2-128 | 131 | 0.219 | 0.211 | **−0.008** |
| ESM2-480 | 483 | 0.223 | 0.206 | **−0.017** |

With weak features the model cannot identify interface residues locally, so it **must** route signal
across the sparse interface — and the bottleneck over-squashes it; adding cross-chain edges relieves
this and helps. With rich ESM2 features each residue already carries enough evolutionary/structural
context to be classified locally, so the same edges no longer relieve a needed bottleneck and instead
mildly **over-smooth** the two chains together, hurting slightly. The interface bottleneck is therefore
real but *conditional* — a phase transition from an over-squashing-limited to an over-smoothing-limited
regime. The causal experiments below are run in the weak (`onehot`) regime where the bottleneck binds.

---

## Method

**Inference-free PINDER pipeline.** Per system I take the two AlphaFold-DB predicted monomers
(pose-free; pLDDT in the B-factor column) as input and the bound complex only for labels. Each monomer
becomes an intra-chain graph (residues within 8 Å of Cα). Node features are per-residue **ESM2**
embeddings (`esm2_t12_35M_UR50D`, 480-d), pLDDT, contact degree, and a chain indicator. Interface
labels (within 8 Å of the other chain in the holo complex) are transferred onto the predicted residues
by sequence alignment. I use heterodimers and skip degenerate systems (prevalence > 0.6 or < 0.02),
giving **383 train / 243 test** systems on the leakage-free split.

**The bottleneck knob.** The two monomers are joined by a *pose-free* cross-chain edge layer that I
sweep as the causal intervention:

- `none` — disconnected (no cross-chain edges; maximal squashing)
- `knn` — embedding-similarity bipartite edges
- `fa` — fully-adjacent final layer
- `virtual` — a global hub node
- `sdrf` — curvature-targeted rewiring (Balanced-Forman), adding only the few edges that relieve the
  most negatively curved necks

**Predictor.** A 4-layer GIN (64 hidden, LayerNorm + residuals), class-weighted cross-entropy, Adam
(2×10⁻³), 120 epochs. Training is deterministic (TF32 off) and averaged over **10 seeds**. Metrics are
AUPRC / AUROC / ECE and a base-rate-free **skill** = AUPRC − prevalence.

**Two over-squashing diagnostics**, threaded through both a synthetic probe and the real data so the
real-data metric is calibrated against ground truth:

- **Effective resistance** R_eff across the interface — the canonical bottleneck measure.
- **Jacobian sensitivity** ‖∂h_v/∂x_u‖ of a read-out node to a source input — measured both on freshly
  initialised models (an architecture property) and *across the interface on the trained models*.

**Modelled vs. native resistance.** The key design choice is *which* bottleneck to correlate with
failure. Rather than the *native* resistance (holo contact graph, which the model never sees), I make
the **modelled** resistance primary: the effective resistance on the model's own pose-free input graph
between the true-interface residues — the bottleneck the network actually routes through.

**Within-system sweep.** Per system I sweep the modelled cross-chain width k ∈ {1, 2, 4, 8} and,
centering both quantities within each system, correlate resistance against skill — a paired,
high-power design that removes between-system noise (the synthetic probe's logic on real data).

**Synthetic probe.** Two Erdős–Rényi communities (16 nodes each) joined by exactly k cross-edges, with
a B-source label read out at the A nodes so the task *requires* cross-interface flow. As k tightens,
accuracy collapses while resistance rises and the Jacobian vanishes (ρ = −0.97 / +0.97 vs. accuracy),
and depth does not rescue a tight bottleneck.

**Statistics.** 10 seeds, deterministic training, bootstrap confidence intervals, pre-registered
one-sided Spearman tests, and a minimum-detectable-effect (MDE = 0.16 at n = 243) so weak correlations
are read as bounded, not headline.

---

## Hypotheses and results

| | prediction | observed | verdict |
|---|---|---|---|
| **H0** | rewiring benefit depends on feature richness | ΔAUPRC flips +0.086 → −0.017 | **supported (crossover)** |
| **H1** | skill ↓ as modelled resistance ↑ | ρ = −0.49 cross-system; −0.13 within-system | **supported** |
| **H2** | rewiring recovers accuracy (causal) | all rewirings > `none`; Jacobian relieved | **supported (causal)** |
| **H3** | topology beats AlphaFold's confidence | modelled −0.47 vs. pLDDT 0.075 | **supported** |
| **H4** | flexibility / N_eff compete | neither predicts failure | negative (informative) |
| **H5** | link holds on the test split | monotone test tertiles, gated on H1 | **supported** |

**H1 — the modelled bottleneck predicts failure.** Cross-system, per-system skill anti-correlates with
the modelled resistance at ρ = **−0.49** (one-sided p = 2×10⁻¹⁶, 95% CI [−0.59, −0.39], sign-stable
across all 10 seeds), whereas the pose-unseen *native* resistance stays weak (ρ = −0.11, p = 0.075) —
as it must, since the model never sees the holo geometry. The within-system sweep confirms it at high
power: ρ = **−0.13** (p = 1.3×10⁻⁵, 95% CI [−0.19, −0.07], 972 system–k points), skill falling as each
system's own bottleneck tightens in 57% of systems.

**H2 — cross-chain rewiring is causal and relieves squashing** (weak `onehot` regime, 10 seeds):

| rewiring | AUPRC | AUROC | ΔAUPRC | ECE |
|---|---|---|---|---|
| none | 0.194 | 0.528 | — | 0.084 |
| knn | 0.232 | 0.589 | +0.038 | 0.039 |
| fa | 0.224 | 0.566 | +0.030 | 0.048 |
| sdrf | 0.230 | 0.584 | +0.035 | 0.040 |
| virtual | **0.233** | **0.589** | **+0.039** | **0.038** |

`knn`, `fa`, and `sdrf` add cross edges *without* a global hub, so their gains over `none` isolate
bottleneck relief rather than the virtual node's added capacity. The curvature-targeted `sdrf` matches
the others adding only ~48 edges per system. The mechanism is confirmed directly on the trained models:
the cross-interface Jacobian is identically **0** for the chain-disconnected `none` (maximal squashing)
and rises to 4.8×10⁻² (`knn`), 5.8×10⁻² (`fa`) once cross edges carry signal.

**H3 / H4 — topology beats confidence.** In partial Spearman correlations, the modelled-resistance term
(ρ = −0.47, p = 5.6×10⁻¹⁵) is an order of magnitude larger than the strongest confidence term, monomer
pLDDT (|ρ| = 0.075), and dwarfs flexibility and co-evolution N_eff (both |ρ| < 0.06). One honest caveat:
AlphaFold-Multimer's complex confidence (ipTM/PAE), the fairer contest, was not exposed by PINDER, so
the confidence baseline is the weaker monomer pLDDT.

**H5 — generalisation gradient.** Stratifying *test* skill by modelled-resistance tertile gives a
monotone decline (0.073 → 0.035 → 0.025), counted as supported because the powered continuous H1 on the
same split is itself significant.

---

## Limitations

The absolute predictor is weak (pooled test AUROC 0.589, per-system skill ≈ +0.04): the pose-free
monomer task is hard, and the claims rest on *relative* effects whose sizes exceed their 10-seed
standard deviation, not on a strong stand-alone classifier. The remaining native-vs-modelled
dissociation is principled rather than contradictory — with pose-free inputs the model's cross-chain
edges are similarity candidates, not the native interface geometry, so per-system difficulty need not
track the native bottleneck even though the modelled one does and relieving it helps causally.

---

## Contents

```
AlphaFold_oversquashing_PINDER.ipynb   the notebook, run end-to-end (383 train / 243 test, 10 seeds)
results/                               outputs of that run
documentation/report.pdf               the 5-page report
```

`results/`:

- `key_numbers.json` — every headline number (feature sweep, H1–H5, Jacobian, MDE)
- `real_findings.md` — narrative findings with all statistics
- `real_model_table.csv` / `h2_table.tex` — the H2 rewiring comparison (AUPRC / AUROC / ECE)
- `per_system.csv` — per-system predictors and skill (n = 243)
- `within_system_sweep.csv` — the within-system bottleneck sweep (972 system–k rows)
