#!/usr/bin/env python3
"""
admet_ai_client.py — predict ADMET properties for molecules from SMILES locally
using ADMET-AI (https://github.com/swansonk14/admet_ai), a Chemprop v2 model ensemble.

Runs entirely on-device — no network call, no flaky server. The 13 MB of Chemprop v2
model weights ship inside the `admet-ai` pip wheel, so `pip install admet-ai` is all
that's needed to get the models.

Usage:
    python3 admet_ai_client.py --input molecules.smi --output admet.csv
    python3 admet_ai_client.py --input data.csv --smiles-column smiles --output admet.csv
    python3 admet_ai_client.py --smiles "CCO" "CC(=O)Oc1ccccc1C(=O)O" --output admet.csv

Device:
    --device cpu (default) | mps | auto
    On Apple Silicon, MPS is supported but is *slower* than CPU here because the models
    are tiny (~1.3 MB each) and MPS dispatch/transfer overhead dominates the matmuls.
    Benchmarks on this machine: CPU ~3.5 ms/mol, MPS ~4.5 ms/mol at n=1000. CPU is
    recommended unless you have a CUDA GPU.

Output: one row per molecule. Columns = input SMILES, then 52 ADMET endpoints, then
52 `_drugbank_approved_percentile` columns (each endpoint's value ranked against the
DrugBank approved reference set). 105 columns total. Endpoint groups: Physicochemical,
Absorption, Distribution, Excretion, Metabolism, Toxicity. Classification endpoints are
probabilities in [0,1]; regression endpoints are predicted values in the endpoint's units.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


def load_smiles(path: str | os.PathLike, smiles_column: str | None = None) -> list[str]:
    """Load SMILES from a .smi/.txt (one per line) or a .csv/.xlsx (column of SMILES)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".smi", ".txt", ".tsv"):
        with open(path) as f:
            return [line.split()[0] for line in f if line.strip() and not line.startswith("#")]
    if suffix == ".csv":
        import csv as _csv
        with open(path, newline="") as f:
            reader = _csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"{path}: empty CSV")
            col = smiles_column or next(
                (c for c in reader.fieldnames if c.lower() in ("smiles", "smile", "canonical_smiles")),
                reader.fieldnames[0],
            )
            return [row[col] for row in reader if row.get(col)]
    if suffix in (".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_excel(path)
        col = smiles_column or next(
            (c for c in df.columns if str(c).lower() in ("smiles", "smile", "canonical_smiles")),
            df.columns[0],
        )
        return [str(v) for v in df[col].tolist() if v]
    raise ValueError(f"unsupported input extension: {suffix}")


def resolve_device(device: str) -> str:
    """Map --device (cpu|mps|auto) to a Lightning accelerator string."""
    if device == "auto":
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return device


def filter_valid_smiles(smiles_list: Sequence[str]) -> tuple[list[str], list[tuple[int, str]]]:
    """Drop blank/invalid SMILES, returning (valid, [(orig_index, bad_smiles)]).

    RDKit-prefilters so a single malformed SMILES doesn't crash the Chemprop batch.
    """
    from rdkit import Chem
    valid: list[str] = []
    bad: list[tuple[int, str]] = []
    for i, s in enumerate(smiles_list):
        s = (s or "").strip()
        if not s:
            bad.append((i, "<blank>"))
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            bad.append((i, s))
            continue
        valid.append(s)
    return valid, bad


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Predict ADMET properties locally with ADMET-AI.")
    p.add_argument("--input", help="path to .smi/.txt/.csv/.xlsx with SMILES")
    p.add_argument("--smiles-column", help="column name for SMILES in a .csv/.xlsx")
    p.add_argument("--smiles", nargs="*", help="one or more SMILES strings on the CLI")
    p.add_argument("--output", "-o", default="admet_results.csv", help="output CSV path")
    p.add_argument(
        "--device",
        choices=("cpu", "mps", "auto"),
        default="cpu",
        help="Lightning accelerator (default cpu; mps is supported but slower for these tiny models)",
    )
    p.add_argument(
        "--no-percentiles",
        action="store_true",
        help="drop the 52 *_drugbank_approved_percentile columns from the output",
    )
    p.add_argument(
        "--no-physchem",
        action="store_true",
        help="exclude computed physicochemical properties (MW, logP, TPSA, alerts, ...)",
    )
    args = p.parse_args(argv)

    if args.input:
        smiles = load_smiles(args.input, args.smiles_column)
    elif args.smiles:
        smiles = list(args.smiles)
    else:
        p.error("provide --input or --smiles")

    if not smiles:
        p.error("no SMILES found in input")

    from admet_ai import ADMETModel

    device = resolve_device(args.device)
    valid, bad = filter_valid_smiles(smiles)
    for idx, s in bad:
        print(f"WARNING: dropping invalid SMILES at input row {idx}: {s!r}", file=sys.stderr)
    if not valid:
        print("ERROR: no valid SMILES to predict after filtering", file=sys.stderr)
        return 1

    print(
        f"Loading ADMET-AI (device={device}); predicting {len(valid)}/{len(smiles)} molecule(s) ...",
        file=sys.stderr,
    )
    model = ADMETModel(include_physchem=not args.no_physchem)
    model.device = device  # ADMETModel only auto-selects cuda; override for mps/cpu.

    try:
        preds = model.predict(smiles=valid)  # DataFrame, aligned to `valid` order
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the CLI user
        print(f"ERROR: prediction failed: {exc}", file=sys.stderr)
        return 1

    # Prepend the input SMILES so each row is self-describing (ADMET-AI omits them).
    import pandas as pd

    out = preds.copy()
    out.insert(0, "smiles", list(valid))

    if args.no_percentiles:
        pct_cols = [c for c in out.columns if c.endswith("_drugbank_approved_percentile")]
        out = out.drop(columns=pct_cols)

    out.to_csv(args.output, index=False)
    print(
        f"Wrote {len(out)} molecule(s) x {out.shape[1]} columns -> {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())