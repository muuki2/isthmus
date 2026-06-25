# Foldable but not dockable

An over-squashing account of AlphaFold's protein–protein interface failures on PINDER.
GRL mini-project (TU Wien).

AlphaFold folds each chain confidently but often mis-places the *complex*. I ask whether this is a
graph **over-squashing** effect: the interface is a sparse inter-chain bottleneck a message-passing
GNN has to route through, while folding each chain is a bottleneck-free intra-chain problem. I study
AlphaFold's interface sub-task — predicting which residues sit at the interface — with my own GNN, on
PINDER's leakage-free split, using the predicted monomers (+pLDDT) as input and the bound complex only
for labels. No folding inference anywhere.

**Main finding.** Over-squashing here is contingent on how rich the node features are. Holding the
model fixed and only varying the per-residue features from weak (degree+pLDDT, one-hot) to rich (ESM2),
the benefit of cross-chain rewiring ΔAUPRC(knn−none) flips sign — +0.086 (baseline) → +0.036 (one-hot)
→ +0.001 → −0.008 → −0.017 (ESM2). Rewiring relieves a real bottleneck when features are weak, but
over-smooths once the features are rich enough to solve the interface locally.

## Contents

- `AlphaFold_oversquashing_PINDER.ipynb` — the notebook, run end-to-end (383 train / 243 test, 10 seeds).
- `results/` — outputs of that run: `key_numbers.json`, `real_findings.md`, the H2 table, and the
  per-system / within-system CSVs.
- `documentation/report.pdf` — the 5-page report.
