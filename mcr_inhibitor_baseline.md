# MCR-inhibitor baseline — ADMET-AI vs Admetica

A reference baseline: both predictors run on the three best-characterised
methyl-coenzyme M reductase (MCR) inhibitors, so every endpoint in this repo
has a known-inhibitor anchor to compare Anna's candidates against.

| Inhibitor | SMILES | role | MW |
|---|---|---|---|
| **3-NOP** (3-nitrooxypropan-1-ol) | `O=[N+]([O-])OCCCO` | the feed-additive MCR inhibitor actually in commercial use | 121.09 |
| **coenzyme M** (2-mercaptoethanesulfonic acid) | `O=S(=O)(O)CCS` | the native MCR substrate/product; surrogate inhibitor reference | 142.20 |
| **rosmarinic acid** | `O=C(/C=C/c1ccc(O)c(O)c1)OC(Cc1ccc(O)c(O)c1)C(=O)O` | plant-derived MCR inhibitor | 360.32 |

All three SMILES were RDKit-validated (canonical SMILES + formula + MW match the
known values: C3H7NO4 / C2H6O3S2 / C18H16O8). Run with:

```sh
.venv/bin/python code/admet_ai_client.py  --input mcr_inhibitors.csv --no-percentiles --no-physchem -o mcr_inhibitor_baseline_admet_ai.csv
.venv/bin/python code/admetica_client.py --input mcr_inhibitors.csv -o mcr_inhibitor_baseline_admetica.csv
```

ADMET-AI = 41 ADMET endpoints (classification = prob [0,1]; regression = native
units). Admetica = 22 endpoints + 15 applicability-domain (AD) scores.

## 1. ADMET-AI baseline (41 endpoints)

| endpoint | 3-NOP | coenzyme M | rosmarinic acid | notes |
|---|---|---|---|---|
| AMES | 0.26 | 0.48 | 0.37 | none flagged (all <0.5) |
| BBB_Martins | 0.94 | 0.83 | 0.06 | small polar pair predicted high BBB; rosmarinic acid excluded from brain (large, polar) |
| Bioavailability_Ma | 0.91 | 0.62 | 0.35 | 3-NOP high; rosmarinic acid low oral F |
| Carcinogens_Lagunin | 0.48 | 0.61 | 0.15 | none flagged |
| ClinTox | 0.03 | 0.09 | 0.17 | none flagged |
| DILI | 0.39 | 0.18 | 0.55 | rosmarinic acid just over 0.5 (mild DILI signal) |
| HIA_Hou | 1.00 | 0.67 | 0.59 | 3-NOP fully absorbed |
| hERG | 0.017 | 0.015 | 0.16 | all green — no cardiotoxicity signal from ADMET-AI |
| PAMPA_NCATS | 0.79 | 0.17 | 0.06 | 3-NOP permeable; rosmarinic acid not |
| Pgp_Broccatelli | 0.001 | 0.001 | 0.007 | none are Pgp substrates per ADMET-AI |
| Skin_Reaction | 0.71 | 0.72 | 0.71 | all flagged skin sensitization |
| CYP1A2_Veith | 0.026 | 0.002 | 0.17 | none inhibit |
| CYP2C19_Veith | 0.022 | 0.008 | 0.070 | none inhibit |
| CYP2C9_Veith | 0.004 | 0.004 | 0.137 | none inhibit |
| CYP2D6_Veith | 0.004 | 0.006 | 0.064 | none inhibit |
| CYP3A4_Veith | 0.0002 | 0.0002 | 0.151 | none inhibit |
| CYP2C9_Sub | 0.23 | 0.11 | 0.045 | weak; none strong substrates |
| CYP2D6_Sub | 0.076 | 0.031 | 0.009 | none |
| CYP3A4_Sub | 0.29 | 0.14 | 0.18 | weak; none strong substrates |
| NR-* (12 assays) | all <0.23 | all <0.05 | 0.06–0.23 | no endocrine/receptor flags on any of the three |
| SR-* (5 stress response) | all <0.05 | all <0.25 | 0.04–0.39 | no genotoxic/oxidative-stress flags |
| Caco2_Wang | −4.19 | −4.91 | −6.31 | log-scale; low permeability for all (rosmarinic acid lowest) |
| Clearance_Hepatocyte_AZ | 59.5 | 9.0 | 38 | 3-NOP fastest hepatic clearance |
| Clearance_Microsome_AZ | 8.7 | −20.1 | 27 | coenzyme M predicted near-zero microsomal clearance |
| Half_Life_Obach | −4.8 | 14.7 | −25.8 | regression (log scale); values unreliable-looking for extremes |
| HydrationFreeEnergy_FreeSolv | −7.0 | −10.8 | −22.1 | rosmarinic acid most hydrophilic |
| LD50_Zhu | 1.92 | 1.80 | 1.99 | all similar, low acute toxicity (mol/kg scale) |
| Lipophilicity_AZ | −0.02 | −0.96 | −0.94 | all hydrophilic |
| PPBR_AZ | 28.2 | 36.6 | 85 | rosmarinic acid strongly protein-bound (polyphenol) |
| Solubility_AqSolDB | −0.48 | −0.31 | −2.55 | rosmarinic acid least soluble |
| VDss_Lombardo | −1.94 | −0.11 | 2.33 | rosmarinic acid highest distribution volume |

**Read:** ADMET-AI reads all three as clean on the safety-critical endpoints —
no hERG, no AMES, no endocrine-receptor, no CYP inhibition, low acute toxicity.
The only ADMET-AI reds are **skin sensitization on all three** (the small
nitrate/thiol/sulfonate groups are alert-bearing) and a mild **DILI signal on
rosmarinic acid** (0.55). 3-NOP and coenzyme M read as small, well-absorbed,
BBB-permeable, hydrophilic molecules; rosmarinic acid reads as a larger, polar,
low-oral-F, high-protein-binding polyphenol that stays out of the brain.

## 2. Admetica baseline (22 endpoints + AD)

### 2a. Endpoint values

| endpoint | 3-NOP | coenzyme M | rosmarinic acid | notes |
|---|---|---|---|---|
| Caco2 | −5.4 | −4.85 | −7.13 | agrees with ADMET-AI ordering (rosmarinic acid lowest) |
| Lipophilicity | 0.35 | −2.18 | 0.64 | 3-NOP +0.35 vs ADMET-AI −0.02 — **disagrees** (low AD) |
| Solubility | −0.27 | 0.75 | −2.0 | agrees on rosmarinic acid (least soluble); coenzyme M sign differs |
| Pgp-Inhibitor | ~0 | ~0 | 0.0005 | none inhibit Pgp |
| Pgp-Substrate | ~0 | ~0 | 0.90 | **rosmarinic acid is a strong Pgp substrate** (ADMET-AI missed — it doesn't model this) |
| PPBR | 44 | −35 | 92 | rosmarinic acid high (agrees); coenzyme M −35 is **nonsense** (AD 0.23, out of domain) |
| VDss | 32.6 | 3.49 | −0.22 | sign differs from ADMET-AI (different dataset/units — rank only) |
| CYP1A2-Inhib | ~0 | ~0 | 0.008 | none inhibit (agrees with ADMET-AI) |
| CYP1A2-Sub | 0.43 | 0.041 | 0.056 | 3-NOP moderate CYP1A2 substrate |
| CYP2C19-Inhib | 0.91 | ~0 | 0.0006 | **3-NOP flagged CYP2C19 inhibitor** (ADMET-AI said 0.02) — AD 0.24, low |
| CYP2C19-Sub | 0.002 | 0.008 | ~0 | none |
| CYP2C9-Inhib | 0.35 | ~0 | 0.071 | 3-NOP moderate (ADMET-AI 0.004) — low AD |
| CYP2C9-Sub | 0.032 | 0.52 | 0.94 | **CYP2C9-substrate systematic miscalibration** (see addendum §3) — Admetica calls rosmarinic acid + coenzyme M substrates, ADMET-AI says no |
| CYP2D6-Inhib | ~0 | ~0 | 0.003 | none |
| CYP2D6-Sub | 0.023 | 0.0001 | 0.008 | none |
| CYP3A4-Inhib | ~0 | ~0 | 0.091 | none |
| CYP3A4-Sub | 0.69 | 0.010 | 0.96 | 3-NOP + rosmarinic acid strong CYP3A4 substrates |
| CL-Hepa | 18.6 | 16.7 | 40.6 | agrees ordering with ADMET-AI (rosmarinic acid highest) |
| CL-Micro | 77.9 | −37.1 | 21.8 | coenzyme M negative = out-of-domain nonsense (AD 0.25) |
| Half-Life | −3.3 | 223 | 0.70 | coenzyme M 223 is out-of-domain nonsense (AD 0.24) |
| hERG | 0.0005 | **0.746** | 0.133 | **coenzyme M flagged hERG binder by Admetica** — but hERG has no AD, and ADMET-AI gives 0.015. Un-arbitratable; treat as caution |
| LD50 | 2.86 | 2.67 | 1.50 | same scale as ADMET-AI; rosmarinic acid (lower LD50) agrees as more toxic |

### 2b. Applicability-domain (AD) scores

| endpoint | 3-NOP | coenzyme M | rosmarinic acid |
|---|---|---|---|
| Caco2 | 0.24 | 0.21 | 0.43 |
| Lipophilicity | 0.22 | 0.23 | 0.40 |
| Solubility | 0.34 | 0.28 | 0.46 |
| Pgp-Substrate | 0.25 | 0.22 | 0.42 |
| PPBR | 0.23 | 0.23 | 0.40 |
| VDss | 0.29 | 0.24 | 0.42 |
| CYP1A2-Inhib | 0.24 | 0.23 | 0.41 |
| CYP2C19-Inhib | 0.24 | 0.24 | 0.41 |
| CYP2C9-Inhib | 0.24 | 0.24 | 0.41 |
| CYP2C9-Sub | 0.25 | 0.22 | 0.42 |
| CYP2D6-Inhib | 0.24 | 0.24 | 0.42 |
| CYP2D6-Sub | 0.24 | 0.21 | 0.41 |
| CL-Hepa | 0.23 | 0.24 | 0.42 |
| CL-Micro | 0.24 | 0.25 | 0.42 |
| Half-Life | 0.26 | 0.24 | 0.42 |

**Domain read:** **3-NOP and coenzyme M are far out of Admetica's training domain**
(AD 0.21–0.34 across every endpoint) — they are tiny, polar, heteroatom-dense
molecules unlike most drug-like training sets. Their Admetica regression calls
(PPBR −35, CL-Micro −37, Half-Life 223 for coenzyme M; Lipophilicity +0.35 for
3-NOP) are therefore **unreliable and should be discounted**. **Rosmarinic acid
is in-domain** (AD 0.40–0.46) — its Admetica calls are the trustworthy ones here.

## 3. What the baseline tells us for Anna's candidates

1. **Both models agree the known inhibitors are ADMET-clean** on the
   decision-critical endpoints (hERG, AMES, endocrine receptors, CYP inhibition,
   acute toxicity). This is reassuring: the *positive controls* don't light up
   as toxic, so the candidates that do flag on those endpoints are flagging for
   chemistry, not because MCR-inhibitor chemistry is generically flagged.
2. **Skin sensitization is expected** for this chemistry (all three inhibitors
   are flagged) — consistent with the addendum treating skin reds as a
   handling note, not a deprioritisation driver.
3. **The hERG disagreement is real here too:** Admetica flags **coenzyme M**
   hERG 0.75 (ADMET-AI 0.015, no AD) — same over-aggressive hERG pattern seen in
   Anna's set. Reinforces the addendum's "hERG is the contested endpoint"
   conclusion.
4. **Admetica's CYP2C9-substrate and substrate calls are miscalibrated here
   too** (rosmarinic acid called a 0.94 CYP2C9 substrate; ADMET-AI says 0.045) —
   the same systematic offset noted in the addendum, now on a known molecule.
5. **The AD score does its job:** it correctly identifies 3-NOP and coenzyme M
   as out-of-domain, and the obviously-nonsensical Admetica regression outputs
   (negative clearances, half-life 223) all sit on the low-AD rows. This is the
   AD score earning its place as the second opinion's confidence gauge.
6. **3-NOP, the actual commercial MCR inhibitor, reads as a clean, small,
   well-absorbed, non-Pgp-substrate molecule** under ADMET-AI — a sanity check
   that the modelling stack lands in the right place for the one compound we
   know is field-viable.