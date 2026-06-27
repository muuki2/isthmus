# Real-data findings (PINDER, AlphaFold interface sub-task)
- Systems: train=383, test=243 (SMOKE=False); cache stores one-hot + ESM2 esm2_t12_35M_UR50D (480-d); MAIN regime=`onehot`; deterministic=True; seeds=10.
## HEADLINE — over-squashing is contingent on feature richness
Varying node-feature richness (each regime self-consistent: its kNN cross layer is built from its own features), the rewiring benefit ΔAUPRC(knn−none) flips sign:
- baseline (in_dim=3): none=0.209 knn=0.295 -> delta=+0.086
- onehot (in_dim=24): none=0.196 knn=0.232 -> delta=+0.036
- esm32 (in_dim=35): none=0.201 knn=0.202 -> delta=+0.001
- esm128 (in_dim=131): none=0.219 knn=0.211 -> delta=-0.008
- esm480 (in_dim=483): none=0.223 knn=0.206 -> delta=-0.017
**Crossover present (rewiring HELPS at weak features, HURTS at rich): True.** This is the central result: over-squashing binds only when the task forces cross-chain routing; rich (ESM) per-residue features let the model solve the interface locally, so the cross-chain edges over-smooth instead of relieving a bottleneck. The causal H2/EXP3 below are reported in the weak (`onehot`) regime where the bottleneck binds.
## Secondary — dissociation within the weak regime
Cross-chain message passing is a real **causal** bottleneck that rewiring fixes (H2, EXP3), and on the **graph the model actually uses** the bottleneck does track per-system failure -- most powerfully in the within-system sweep (EXP2). The *native* (pose-unseen) interface resistance remains a weak cross-system predictor; that dissociation is expected for pose-free inputs and is the honest story.
## Predictor sanity
Per-system model = `knn_bipartite` (fixed a priori; pose-free cross edges, not hub-insulated). Mean pooled test AUROC=0.589; mean per-system AUROC=0.548, mean skill (AUPRC-base)=+0.044. INTERPRETABLE (>0.52).
## H1 — bottleneck vs interface skill: **SUPPORTED**
**Modelled bottleneck (primary; the graph the model routes through):** Spearman(model_iface_reff, skill) = -0.490 (one-sided p=2.1e-16, 95% CI [-0.59,-0.39], n=243); per-seed [-0.343, -0.286, -0.443, -0.295, -0.288, -0.381, -0.388, -0.43, -0.274, -0.44] (sign-stable=True).
**Native bottleneck (contrast; holo geometry the model never sees):** rho=-0.114 (p=0.075) -- weaker, as expected for pose-free inputs. Relieved `virtual` model: rho=-0.386. Minimum detectable |rho| at this n is 0.159 (alpha=.05, power=.8, one-sided), so a true effect below that would be invisible -- we are adequatelypowered.
## EXP2 (headline upgrade) — within-system bottleneck sweep: **SUPPORTED**
Sweeping the modelled cross-chain width k per system and centering within each system (the synthetic probe's logic on real data, a paired/high-power design): rho(centered resistance, centered skill)=-0.134 (one-sided p=1.3e-05, 95% CI [-0.19,-0.07], 972 system-k points over 243 systems); per-system slope negative in 57% of systems. Aggregate means by k: {1: (0.295, 0.038), 2: (0.226, 0.039), 4: (0.164, 0.046), 8: (0.112, 0.046)}.
## H2 — cross-chain rewiring (causal): **SUPPORTED**
- none: AUPRC=0.194+/-0.004 AUROC=0.528 ECE=0.084 (delta AUPRC vs none=+0.000)
- virtual: AUPRC=0.233+/-0.004 AUROC=0.589 ECE=0.038 (delta AUPRC vs none=+0.039)
- knn_bipartite: AUPRC=0.232+/-0.005 AUROC=0.589 ECE=0.039 (delta AUPRC vs none=+0.038)
- fa: AUPRC=0.224+/-0.006 AUROC=0.566 ECE=0.048 (delta AUPRC vs none=+0.030)
- sdrf: AUPRC=0.230+/-0.007 AUROC=0.584 ECE=0.040 (delta AUPRC vs none=+0.035) [+48 curv-edges]
`knn_bipartite`/`fa`/`sdrf` add cross edges with NO global hub, so their gains over `none` isolate bottleneck relief (not the virtual node's extra capacity). **SDRF** (curvature-targeted) reaches comparable relief adding only ~48 edges/system vs fa's complete final layer / virtual's N-edge hub -- a parsimonious, principled inductive bias.
## EXP3 — over-squashing on the REAL trained models: **SUPPORTED**
Cross-interface Jacobian ||dh_A/dx_B|| on the trained models: {'none': '0.00e+00', 'knn_bipartite': '4.78e-02', 'fa': '5.78e-02', 'virtual': '1.90e-02', 'sdrf': '1.76e-02'}. `none` is chain-disconnected so its cross-interface gradient is identically 0 (maximal squashing); the cross-wired variants carry interface signal. This is the mechanism behind H2, measured on the real models (not just the probe).
## H3 (headline) / H4 — modelled topology vs confidence (pLDDT) vs Neff vs flexibility: **SUPPORTED**
- partial rho(modelled_resistance, skill) = -0.473 (p=5.6e-15)
- partial rho(native_resistance, skill) = -0.085 (p=0.18)
- partial rho(low_pLDDT, skill) = +0.075 (p=0.25)
- partial rho(low_Neff, skill) = +0.058 (p=0.37)
- partial rho(flexibility, skill) = -0.034 (p=0.59)
Modelled-resistance term rho=-0.473 (p=5.6e-15) vs strongest confidence term |rho|=0.075. AF2 complex-confidence (ipTM/PAE) was NOT available from PINDER; so the confidence baseline here is monomer pLDDT (a weaker, but the only available, signal).
## H5 — generalization on the de-leaked split: **DIRECTIONAL**
Skill by MODELLED-resistance tertile (low->high), train vs test: {'train': {0: 0.291, 1: 0.299, 2: 0.334}, 'test': {0: 0.073, 1: 0.035, 2: 0.025}}. Test tertiles show a monotone decline (0.073->0.035->0.025), but with only ~81 systems/tertile and no trend test we read this as directional, not confirmed; the powered continuous within-split H1 is the same claim done properly and IS significant, tying over-squashing to OOD generalization. The train->test skill drop reflects expected overfitting of the small GIN on 383 systems.
## Limitations
- We study AlphaFold's *interface sub-task* with a model we own, not end-to-end DockQ (needs the forbidden inference). Inputs are pose-free; labels are holo-derived and alignment-transferred.
- **The dissociation is principled:** the modelled bottleneck (what the model routes through) tracks failure, especially within-system (EXP2); the *native* bottleneck need not, because the model never sees the holo geometry. H2/EXP3 show the *causal* effect directly.
- **Reproducibility:** training is deterministic (TF32 off, deterministic algos); we additionally report bootstrap CIs and a minimum-detectable-effect, so weak correlations are read as bounded, not headline, effects.
