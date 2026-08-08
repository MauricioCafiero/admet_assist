#!/usr/bin/env python3
"""
max_dosage.py — derive a maximum-dosage estimate from predicted LD50.

Reads the LD50 columns already produced by admet_ai_client.py (column LD50_Zhu)
and/or admetica_client.py (column LD50) and adds derived dose columns. No model
is loaded — this is pure post-processing of the ADMET result CSVs.

Both LD50 endpoints are rat oral LD50 on the scale log10(1/(mol/kg)) (ADMET-AI
states this in its endpoint metadata; Admetica is trained on the same TDC Zhu
dataset, and the repo baseline confirms the two agree on scale). So:

    LD50_mol_per_kg = 10 ** (-v)                    # v = predicted LD50 value
    LD50_mg_per_kg  = LD50_mol_per_kg * MW * 1000   # MW in g/mol, from RDKit

Derived doses:
    NOAEL_mg_per_kg = LD50_mg_per_kg / 10            # standard LD50->NOAEL heuristic (fixed)
    MTD_mg_per_kg   = LD50_mg_per_kg / safety_factor # --safety-factor, default 10

Feed-additive scaling (the user supplies animal body weight and daily feed
intake — defaults are cattle):
    MTD_mg_per_head_per_day = MTD_mg_per_kg * body_weight
    MTD_mg_per_kg_feed      = MTD_mg_per_head_per_day / feed_intake   # inclusion rate

WARNING: the Zhu LD50 model is weak (R^2 ~ 0.60, MAE ~ 0.45 log units, per
ADMET-AI metadata). Every number below is a SCREENING RANK-ORDER ESTIMATE, not
a real toxicology value — do not use it to set an actual administered dose.

Usage:
    python3 max_dosage.py --admet-ai admet.csv --admetica admetica.csv -o dosage.csv
    python3 max_dosage.py --admet-ai admet.csv --body-weight 500 --feed-intake 20 -o dosage.csv
    python3 max_dosage.py --admet-ai admet.csv --safety-factor 3 -o dosage.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

# Source -> column name holding the predicted LD50 value in that source's CSV.
LD50_COLUMN = {
    "admet_ai": "LD50_Zhu",
    "admetica": "LD50",
}

NOAEL_FACTOR = 10  # NOAEL ~= LD50/10 (fixed heuristic)


def canonicalize(smiles: str) -> str | None:
    """RDKit canonical SMILES, or None if the SMILES is blank/invalid."""
    from rdkit import Chem
    s = (smiles or "").strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol) if mol is not None else None


def mw_from_canonical(canonical: str) -> float:
    """Molecular weight (g/mol) from a canonical SMILES."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    return float(Descriptors.MolWt(Chem.MolFromSmiles(canonical)))


def ld50_to_mg_per_kg(value: float, mw: float) -> float:
    """Convert a predicted LD50 (log10(1/(mol/kg))) to mg/kg using MW (g/mol)."""
    return (10.0 ** (-float(value))) * mw * 1000.0


def load_ld50_column(path: str | os.PathLike, source: str) -> dict[str, float]:
    """Read a source's ADMET result CSV -> {canonical_smiles: ld50_value}.

    Invalid/blank SMILES and missing/NaN LD50 values are dropped.
    """
    import pandas as pd

    col = LD50_COLUMN[source]
    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        raise ValueError(f"{path}: no 'smiles' column (run {source}_client.py to produce it)")
    if col not in df.columns:
        raise ValueError(f"{path}: no {col!r} column (expected from {source}_client.py)")

    out: dict[str, float] = {}
    for raw, v in zip(df["smiles"], df[col]):
        c = canonicalize(raw)
        if c is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        import math
        if math.isnan(f):
            continue
        out[c] = f
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Derive a maximum-dosage estimate from predicted LD50 (post-processing).",
    )
    p.add_argument("--admet-ai", metavar="CSV", help="ADMET-AI result CSV with an LD50_Zhu column")
    p.add_argument("--admetica", metavar="CSV", help="Admetica result CSV with an LD50 column")
    p.add_argument("--output", "-o", default="max_dosage.csv", help="output CSV path")
    p.add_argument("--safety-factor", type=float, default=10.0,
                   help="MTD = LD50_mg_per_kg / this (default 10; lower -> less conservative MTD)")
    p.add_argument("--body-weight", type=float, default=500.0,
                   help="animal body weight in kg (default 500, cattle)")
    p.add_argument("--feed-intake", type=float, default=20.0,
                   help="kg of feed consumed per animal per day (default 20, cattle dry matter)")
    args = p.parse_args(argv)

    sources: list[tuple[str, str]] = []
    if args.admet_ai:
        sources.append(("admet_ai", args.admet_ai))
    if args.admetica:
        sources.append(("admetica", args.admetica))
    if not sources:
        p.error("provide --admet-ai and/or --admetica CSV")

    # Silence RDKit's parse warnings on the (already-validated) input SMILES.
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    per_source: dict[str, dict[str, float]] = {}
    for name, path in sources:
        try:
            per_source[name] = load_ld50_column(path, name)
        except Exception as exc:  # noqa: BLE001 - surface a clean error, keep going
            print(f"ERROR: could not read {name} CSV {path!r}: {exc}", file=sys.stderr)
            return 1
        print(f"Read {len(per_source[name])} molecule(s) from {name} ({path})", file=sys.stderr)

    # Union of molecules across the provided sources, joined on canonical SMILES.
    all_cano = list(dict.fromkeys(c for src in per_source.values() for c in src))

    import math
    rows: list[dict[str, float | str]] = []
    for c in all_cano:
        mw = mw_from_canonical(c)
        row: dict[str, float | str] = {"smiles": c, "MW": round(mw, 2)}
        for name in (n for n, _ in sources):
            v = per_source[name].get(c)
            if v is None:
                continue
            ld50_mg_kg = ld50_to_mg_per_kg(v, mw)
            noael_mg_kg = ld50_mg_kg / NOAEL_FACTOR
            mtd_mg_kg = ld50_mg_kg / args.safety_factor
            mtd_mg_head_day = mtd_mg_kg * args.body_weight
            mtd_mg_kg_feed = mtd_mg_head_day / args.feed_intake
            pfx = f"{name}__"
            row[pfx + "LD50_pred"] = v
            row[pfx + "LD50_mg_kg"] = round(ld50_mg_kg, 1)
            row[pfx + "NOAEL_mg_kg"] = round(noael_mg_kg, 1)
            row[pfx + "MTD_mg_kg"] = round(mtd_mg_kg, 1)
            row[pfx + "MTD_mg_head_day"] = round(mtd_mg_head_day, 1)
            row[pfx + "MTD_mg_kg_feed"] = round(mtd_mg_kg_feed, 1)
        rows.append(row)

    import pandas as pd
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(
        f"Wrote {len(out)} molecule(s) x {out.shape[1]} columns "
        f"(BW={args.body_weight} kg, feed={args.feed_intake} kg/day, "
        f"safety_factor={args.safety_factor}) -> {args.output}",
        file=sys.stderr,
    )
    print(
        "WARNING: LD50 model is weak (R^2~0.60); these are rank-order screening "
        "estimates, not administered doses.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())