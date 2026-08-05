#!/usr/bin/env python3
"""
make_addendum.py — build an *addendum* to anna_admet_report.md that folds the
Admetica cross-check into the ADMET-AI ranking: a revised ranked list with
Admetica data at the top, then "where Admetica might change recommendations",
then "new information Admetica adds", then a condensed mesh-evidence section.

Reads:
  anna_top_15.csv   (SMILES + IUPAC Name, in the per-compound order of the main report)
  anna_admet_ai.csv (ADMET-AI predictions, aligned by SMILES)
  anna_admetica.csv (Admetica predictions + <endpoint>_AD, aligned by SMILES)

The ADMET-AI rank / green / red / affinity / price are carried over from the main
report (hardcoded below, indexed by the CSV row order 1..15) so this stays an
addendum rather than a re-derivation.

Run (from repo root, project .venv):
    .venv/bin/python code/make_addendum.py
    /Users/cafierom/python_mac/moltui/mt-env/bin/python code/make_pdf.py \
        anna_admetica_addendum.md anna_admetica_addendum.pdf
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Per-compound metadata from the main ADMET-AI report, indexed by CSV row (1..15).
# (short_name, rank, green, red, affinity, price_str, orig_rationale)
META = {
    1:  ("m-coumaric acid",                    1, 7, 1, -7.4, "£1.66",   "Best value-for-safety: only DILI red, cheapest, low CNS pen."),
    2:  ("4-(4-hydroxyphenyl)but-3-en-2-one",  6, 6, 1, -7.4, "£1.85",   "Cheap; sole red = skin sensitization (handling, not ingestion)."),
    3:  ("lauric acid",                        3, 5, 2, -7.6, "£3.13",   "Only candidate with in-vivo cattle evidence; reds = high BBB + NIH x2."),
    4:  ("2-nonenoic acid",                    5, 7, 2, -7.9, "£3.48",   "Strong affinity, 4/5 percentiles reinforce; reds = carcinogen signal + BRENK x2."),
    5:  ("isoeugenol",                        13, 5, 1, -6.8, "£11.81",  "Anna severity HIGH — IARC 2B animal carcinogenicity; weakest affinity. Avoid."),
    6:  ("ethyl caffeate",                    12, 3, 6, -7.7, "£12.90", "Most red flags (AMES, DILI, skin, CYP, AhR, BRENK)."),
    7:  ("4-[(4-hydroxyphenyl)methyl]phenol",  7, 8, 3, -7.4, "£19.10", "Most greens; ER endocrine red (bisphenol-F concern) — only if identity confirmed."),
    8:  ("isoferulic acid",                    4, 8, 1, -7.0, "£225",    "Cleanest absolute profile (DILI only red); costly; LD50 undermines."),
    9:  ("beta-bisabolene",                    15, 6, 5, -9.0, "£23,200", "Strong affinity but 5 reds (hERG, skin, BBB, CYP, acute tox) + absurd cost."),
    10: ("bis(4-hydroxyphenyl)penta-dien-3-one", 14, 4, 5, -9.1, "—",    "Strongest affinity but 5 reds (DILI, skin, CYP, ER, acute tox). Needs synth."),
    11: ("(4-hydroxyphenyl) octanoate",        11, 5, 4, -8.8, "—",      "ER endocrine + BRENK x2; needs synthesis."),
    12: ("ethyl coumarate",                    10, 3, 3, -7.8, "—",      "Reds DILI + skin + CYP; needs synthesis."),
    13: ("heptenoic acid",                     2, 8, 2, -7.5, "—",      "Only compound with all 6 green flags percentile-reinforced; reds carcinogen + BRENK."),
    14: ("3,6-dimethylhepta-2,5-dien-1-ol",     8, 7, 2, -7.3, "—",      "4/5 reinforce, weakish affinity; reds = skin + BBB. Needs synth."),
    15: ("1-methoxy-4-(2-methylprop-1-enyl)benzene", 9, 5, 3, -7.3, "—", "Anethole/estragole-family caution; reds BBB + CYP + acute tox."),
}

LOW_AD = 0.30
HIGH_AD = 0.40
HERG_GREEN = 0.10
HERG_RED = 0.50


def herg_flag(p: float) -> str:
    if p >= HERG_RED:
        return "red"
    if p < HERG_GREEN:
        return "green"
    return "—"


def load_mols():
    out = []
    with open(ROOT / "anna_top_15.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            s = (r.get("SMILES") or "").strip()
            n = (r.get("IUPAC Name") or "").strip()
            if s and n:
                out.append((s, n))
    return out


def main():
    import pandas as pd

    mols = load_mols()
    ai = pd.read_csv(ROOT / "anna_admet_ai.csv").set_index("smiles").reindex([s for s, _ in mols]).reset_index()
    ad = pd.read_csv(ROOT / "anna_admetica.csv").set_index("smiles").reindex([s for s, _ in mols]).reset_index()

    def ai_v(i, c):
        try: return float(ai.iloc[i][c])
        except Exception: return float("nan")

    def ad_v(i, c):
        try: return float(ad.iloc[i][c])
        except Exception: return float("nan")

    ad_cols = [c for c in ad.columns if c.endswith("_AD")]

    def min_ad(i):
        vals = [float(ad.iloc[i][c]) for c in ad_cols if float(ad.iloc[i][c]) == float(ad.iloc[i][c])]
        return min(vals) if vals else float("nan")

    # csv_idx 1..15 -> row index 0..14
    def row(idx):
        return idx - 1

    L: list[str] = []
    W = L.append

    W("# Addendum — Admetica cross-check on Anna's top 15")
    W("")
    W("An *addendum* to `anna_admet_report.md`. The main report ranks the 15 "
      "anti-methanogenic candidates using **ADMET-AI** (52-endpoint Chemprop v2 "
      "**5-model ensembles**). This layer runs a second, **independent** predictor — "
      "**Admetica** (Datagrok; 22 per-endpoint Chemprop v2 **single models**) — on the "
      "same 15 molecules and asks three questions: (1) does the ranking move when "
      "Admetica is folded in, (2) where *might* it change a recommendation, and (3) "
      "what genuinely new information does Admetica add.")
    W("")
    W("Two things make Admetica different from just re-running ADMET-AI. First, 15 of "
      "its 22 endpoints carry an **applicability-domain (AD) cosine-similarity score** "
      "(higher = more in-domain, ~0–1) — a per-call confidence check ADMET-AI lacks "
      "(it offers DrugBank percentiles, a *population* comparison, not an in-domain "
      "check). Where the models disagree **and Admetica's AD is high (≥0.40)**, prefer "
      "Admetica; where **AD is low (<0.30)**, prefer ADMET-AI. Second, **7 endpoints have "
      "no AD vector** (P-gp inhibition, CYP3A4 inhib/substrate, hERG, LD50, CYP1A2/2C19 "
      "substrate) — those disagreements *cannot be arbitrated*.")
    W("")
    W("_Read this as a confidence layer, not a tie-breaker: Admetica is a second opinion, "
      "and several of its disagreements (notably hERG and CYP-substrate) look systematic "
      "rather than informative. Details below._")
    W("")

    # ---- 1. Revised ranked list with Admetica folded in --------------------
    W("## 1. Revised ranking — Admetica data folded in")
    W("")
    W("Rank order is **unchanged from the main report**: where Admetica has a usable AD "
      "score it **reinforces** the ADMET-AI safety read on the inhibition endpoints (79% "
      "agreement) and on rank-ordering of solubility/logP/clearance, so it does not "
      "displace any compound. The Admetica columns are added so each row carries the "
      "second-opinion data; the **Δ** column flags the per-compound Admetica signal. "
      "(Tier labels from the main report: 1–3 pursue · 4–6 consider with caveats · "
      "7–9 marginal/niche · 10–15 deprioritise.)")
    W("")
    W("| Rank | Compound | 🟢 | 🔴 | Aff | £/g | min AD | Adm hERG | Admetica Δ | Rationale (Admetica-adjusted) |")
    W("|---|---|---|---|---|---|---|---|---|---|")

    # build Δ + adjusted rationale per csv_idx
    def delta(idx):
        i = row(idx)
        h_ai = herg_flag(ai_v(i, "hERG"))
        h_ad = herg_flag(ad_v(i, "hERG"))
        mad = min_ad(i)
        parts = []
        # hERG disagreement
        if h_ai != h_ad and not (h_ai == "—" and h_ad == "—"):
            parts.append(f"hERG contested (Adm {ad_v(i,'hERG'):.2f}, no AD)")
        # arbitrated CYP inhibition changes (AD >= HIGH_AD)
        for col, label in [("CYP1A2-Inhibitor", "CYP1A2-inh"),
                           ("CYP2C19-Inhibitor", "CYP2C19-inh"),
                           ("CYP2C9-Inhibitor", "CYP2C9-inh")]:
            a = ai_v(i, {"CYP1A2-Inhibitor": "CYP1A2_Veith",
                         "CYP2C19-Inhibitor": "CYP2C19_Veith",
                         "CYP2C9-Inhibitor": "CYP2C9_Veith"}[col])
            d = ad_v(i, col)
            adc = ad_v(i, col + "_AD")
            if adc == adc and adc >= HIGH_AD and (a >= 0.5) != (d >= 0.5):
                if d >= 0.5:
                    parts.append(f"{label} red added (Adm {d:.2f}, AD {adc:.2f})")
                else:
                    parts.append(f"{label} red removed (Adm {d:.2f}, AD {adc:.2f})")
        # new Pgp-substrate flag
        ps = ad_v(i, "Pgp-Substrate")
        psad = ad_v(i, "Pgp-Substrate_AD")
        if ps >= 0.5:
            tag = f"Pgp-substrate {ps:.2f}"
            if psad == psad:
                tag += f" (AD {psad:.2f})"
            parts.append(tag)
        # low AD overall
        if mad == mad and mad < LOW_AD:
            parts.append(f"low-AD ({mad:.2f}) — Adm thin here")
        return "; ".join(parts) if parts else "reinforces ADMET-AI"

    for idx in sorted(META, key=lambda k: META[k][1]):  # by rank
        short, rank, g, r, aff, price, _orig = META[idx]
        i = row(idx)
        mad = min_ad(i)
        mad_s = f"{mad:.2f}" if mad == mad else "—"
        h_ad = ad_v(i, "hERG")
        W(f"| {rank} | {short} | {g} | {r} | {aff} | {price} | {mad_s} | {h_ad:.2f} | {delta(idx)} | {_orig} |")
    W("")
    W("**Bottom line on the ranking:** it does not move. Admetica's high-AD calls agree "
      "with ADMET-AI on the safety-critical inhibition endpoints and on solubility/logP "
      "ordering; its hERG calls can't be arbitrated; its CYP-substrate calls are "
      "systematically different from ADMET-AI's (see §3). So the main report's tiers hold. "
      "The interesting question is the *what-if* on hERG — next.")
    W("")

    # ---- 2. Where Admetica might change recommendations --------------------
    W("## 2. Where Admetica might change recommendations")
    W("")
    W("### 2a. The hERG decision point (the one real lever — but un-arbitrated)")
    W("")
    W("Admetica's hERG model is far more aggressive than ADMET-AI's: it **changes the hERG "
      "call on 12 of 15 compounds** — flagging **8 as hERG binders (≥0.5)** where ADMET-AI "
      "is below threshold (including the **#1 pick m-coumaric acid** and the otherwise-"
      "clean **isoferulic acid #4**), and reading the other 4 as *more* favourable (lauric "
      "acid #3 and 4-[(4-hydroxyphenyl)methyl]phenol #7 gain a green; 4-(4-hydroxyphenyl)"
      "but-3-en-2-one #6 and heptenoic acid #13 lose a green). hERG has **no AD vector**, "
      "so there is no umpire. ADMET-AI's hERG reads are *DrugBank-percentile-reinforced* "
      "(low percentiles for the clean compounds, i.e. greener than the typical approved "
      "drug); Admetica's are a single model with no in-domain check. So the working "
      "position is: **keep ADMET-AI's hERG call, treat Admetica's as an unverified "
      "caution.** But if hERG liability is decision-critical for the use case, this is "
      "the single endpoint worth measuring experimentally, because crediting Admetica "
      "would reshuffle the top of the list:")
    W("")
    W("| # | Compound | rank | ADMET-AI hERG | Admetica hERG | hERG flag flips | rank-direction impact if credited |")
    W("|---|---|---|---|---|---|---|")
    for idx in range(1, 16):
        i = row(idx)
        short, rank, *_ = META[idx]
        a = ai_v(i, "hERG"); d = ad_v(i, "hERG")
        fa, fd = herg_flag(a), herg_flag(d)
        if fa == fd:
            continue
        arrow = f"{fa} → {fd}"
        impact = "—"
        if fa == "green" and fd == "red":
            impact = "**demote** (loses a clean hERG green, gains a red)"
        elif fa == "—" and fd == "red":
            impact = "demote (gains a hERG red)"
        elif fa == "green" and fd == "—":
            impact = "soft demote (loses a hERG green)"
        elif fa == "—" and fd == "green":
            impact = "promote (gains a hERG green)"
        elif fa == "red" and fd == "green":
            impact = "promote (red → green)"
        W(f"| {idx} | {short} | {rank} | {a:.3f} ({fa}) | {d:.3f} ({fd}) | {arrow} | {impact} |")
    W("")
    W("The compounds that would move most are the **clean-tier picks**: **m-coumaric acid "
      "(#1)** and **isoferulic acid (#4)** would each gain a hERG red and likely slip out "
      "of the pursue/consider tiers, while **lauric acid (#3)** and **4-[(4-hydroxyphenyl)"
      "methyl]phenol (#7)** would gain a hERG *green* and strengthen. Again — this is a "
      "**what-if**, not the recommendation: Admetica's hERG calls for this small-polar-"
      "phenolic/acids set look over-calibrated and lack any in-domain check, so crediting "
      "them wholesale is not justified. The honest output is: _hERG is the contested "
      "endpoint; if it matters, test it._")
    W("")
    W("### 2b. Arbitrated changes (Admetica disagrees with AD ≥ 0.40)")
    W("")
    W("These are the calls where the AD score actually lets us pick a winner. They are "
      "real but mostly land on already-deprioritised compounds, so the rank impact is small:")
    W("")
    W("| Compound (rank) | Endpoint | ADMET-AI | Admetica | AD | Verdict | Rank impact |")
    W("|---|---|---|---|---|---|---|")
    arb = []
    for idx in range(1, 16):
        i = row(idx)
        short, rank, *_ = META[idx]
        for col, ai_col, label in [("CYP1A2-Inhibitor", "CYP1A2_Veith", "CYP1A2 inhibition"),
                                  ("CYP2C19-Inhibitor", "CYP2C19_Veith", "CYP2C19 inhibition"),
                                  ("CYP2C9-Inhibitor", "CYP2C9_Veith", "CYP2C9 inhibition")]:
            a = ai_v(i, ai_col); d = ad_v(i, col); adc = ad_v(i, col + "_AD")
            if adc != adc or adc < HIGH_AD:
                continue
            if (a >= 0.5) == (d >= 0.5):
                continue
            verdict = "prefer Admetica (high AD)" if adc >= HIGH_AD else "uncertain"
            if d >= 0.5:
                impact = "adds a red — mild demotion" if rank <= 9 else "no change (already deprioritised)"
            else:
                impact = "removes a red — mild promotion" if rank <= 9 else "no change (already deprioritised)"
            arb.append((rank, short, label, a, d, adc, verdict, impact))
    for rank, short, label, a, d, adc, verdict, impact in sorted(arb):
        W(f"| {short} ({rank}) | {label} | {a:.3f} | {d:.3f} | {adc:.2f} | {verdict} | {impact} |")
    W("")
    W("The one that touches a cleanish compound: **4-(4-hydroxyphenyl)but-3-en-2-one "
      "(rank 6)** — ADMET-AI gives CYP2C19 inhibition 0.122 (green), Admetica 0.651 "
      "with AD 0.43 (decent). That is a second red flag on a compound previously noted "
      "for having only a skin-sensitization red, and is the single arbitrated result "
      "that could justify nudging it down within the 'consider with caveats' tier. "
      "(CYP1A2 reds *removed* from ranks 7 and 14 are favourable but don't rescue them — "
      "#7's ER-endocrine concern and #14's BBB/skin reds still dominate.)")
    W("")
    W("### 2c. Where Admetica does NOT change anything")
    W("")
    W("- **Low-AD compounds** — #4 2-nonenoic acid (min AD 0.28), #13 heptenoic acid "
      "(0.29), #14 3,6-dimethylheptadien-1-ol (0.21): Admetica is extrapolating here, so "
      "its calls are unreliable and cannot undermine the (percentile-reinforced) "
      "ADMET-AI reads. Net: no change, just lower confidence in the second opinion.")
    W("- **Regression safety endpoints** — LD50, solubility, logP rank-order similarly "
      "across the 15 (mean Spearman ρ = 0.48; ρ ≥ 0.7 for solubility, logP, microsome "
      "clearance). VDss anti-correlates (ρ = −0.35), but VDss isn't a driver of the "
      "ranking.")
    W("- **Inhibition endpoints overall** — 83/105 high/low calls agree (79%); the "
      "disagreements are itemised in §2b or are on already-deprioritised compounds.")
    W("")

    # ---- 3. New information Admetica adds ----------------------------------
    W("## 3. New information Admetica adds")
    W("")
    W("Two things ADMET-AI does not provide at all:")
    W("")
    W("### 3a. Three endpoints ADMET-AI doesn't model")
    W("")
    W("CYP1A2-substrate, CYP2C19-substrate and Pgp-substrate. Pgp-substrate carries an AD "
      "score; the two CYP-substrate endpoints do not. For a **feed-additive** context: "
      "**CYP substrate** ≈ hepatic metabolic turnover (bears on residue/persistence — "
      "faster turnover can mean less residue, or less persistent exposure), and "
      "**Pgp-substrate** ≈ efflux-transporter substrate (bears on oral bioavailability). "
      "These are PK flags, not safety reds — interpret with care and don't rank on them.")
    W("")
    W("| # | Compound | CYP1A2-sub | CYP2C19-sub | Pgp-sub | Pgp-sub AD |")
    W("|---|---|---|---|---|---|")
    for idx in range(1, 16):
        i = row(idx)
        short = META[idx][0]
        c1 = ad_v(i, "CYP1A2-Substrate"); c2 = ad_v(i, "CYP2C19-Substrate")
        ps = ad_v(i, "Pgp-Substrate"); psad = ad_v(i, "Pgp-Substrate_AD")
        psad_s = f"{psad:.2f}" if psad == psad else "—"
        W(f"| {idx} | {short} | {c1:.3f} | {c2:.3f} | {ps:.3f} | {psad_s} |")
    W("")
    W("Notable: **4-[(4-hydroxyphenyl)methyl]phenol (#7)** is a strong Pgp substrate "
      "(0.992) — an extra PK concern layered on its endocrine flag. **Isoeugenol (#5)** "
      "and the **anethole-family compound (#15)** are also strong Pgp substrates (≥0.70). "
      "Several compounds are strong **CYP1A2 substrates** (m-coumaric 0.892, isoeugenol "
      "0.978, isoferulic 0.940) — i.e. predicted rapid CYP1A2-mediated metabolism.")
    W("")
    W("_Caveat on the substrate endpoints:_ the **shared** CYP2C9-substrate model is "
      "systematically miscalibrated versus ADMET-AI (0% agreement — Admetica calls nearly "
      "every compound a substrate, ADMET-AI calls nearly none), so treat Admetica's "
      "substrate calls as a *different lens*, not a confirmation of anything.")
    W("")
    W("### 3b. The per-call applicability-domain (AD) confidence score")
    W("")
    W("This is Admetica's main genuine addition. ADMET-AI has nothing equivalent at the "
      "per-call level (its DrugBank percentiles compare a molecule to the approved-drug "
      "population, not to the model's training domain). The AD score says, per call, "
      "whether the model is interpolating or extrapolating. Low-AD calls to discount:")
    W("")
    W("| Compound (rank) | endpoint | AD | Admetica value |")
    W("|---|---|---|---|")
    low = []
    for idx in range(1, 16):
        i = row(idx)
        short, rank, *_ = META[idx]
        for c in ad_cols:
            adv = float(ad.iloc[i][c])
            if adv == adv and adv < LOW_AD:
                ep = c[:-3]  # strip _AD
                low.append((adv, short, rank, ep, float(ad.iloc[i][ep])))
    for adv, short, rank, ep, val in sorted(low):
        W(f"| {short} ({rank}) | {ep} | {adv:.2f} | {val:.3f} |")
    W("")
    W(f"All {len(low)} sub-{LOW_AD} calls cluster on the three low-AD compounds above "
      "(#4, #13, #14) — Admetica is least reliable exactly where its opinion is least "
      "needed (the ranking there already rests on ADMET-AI + Anna's evidence + "
      "percentiles).")
    W("")

    # ---- 4. How the two models mesh (condensed evidence) ------------------
    W("## 4. How the two models mesh (condensed)")
    W("")
    W("| Mesh | result |")
    W("|---|---|")
    W("| Regression (9 shared endpoints) | mean Spearman ρ **0.48**; solubility 0.86, logP 0.83, microsome clearance 0.72 agree well; **VDss anti-correlates (−0.35)** |")
    W("| Classification — inhibition (7 endpoints × 15) | **83/105 agree (79%)** — usable cross-check |")
    W("| Classification — substrate (3 shared) | 12/45 agree; **CYP2C9-substrate 0%** — systematic, not noise |")
    W("| Classification — hERG | 47% agree, **no AD** — can't arbitrate |")
    W("| AD arbiter rule | disagree + AD ≥ 0.40 → prefer Admetica; AD < 0.30 → prefer ADMET-AI; no AD → no umpire |")
    W("")
    W("Translation for the ranking: where Admetica can speak confidently (high AD) it "
      "agrees with ADMET-AI on the endpoints that drive the safety ranking; where it "
      "disagrees it is either un-arbitratable (hERG) or systematic (CYP-substrate) or on "
      "already-deprioritised compounds. Hence the main report's ranking stands, and "
      "Admetica's contribution is the **AD confidence score** and the **three new "
      "substrate endpoints** in §3 — plus a flag that **hERG is the one endpoint worth "
      "measuring** if cardiotoxicity liability is in scope.")
    W("")

    out = ROOT / "anna_admetica_addendum.md"
    out.write_text("\n".join(L) + "\n")
    print(f"Wrote {out}  ({out.stat().st_size} bytes, {len(L)} lines)")


if __name__ == "__main__":
    main()