#!/usr/bin/env python3
"""
admetica_client.py — predict ADMET properties for molecules from SMILES locally,
using the Admetica (https://github.com/datagrok-ai/admetica) per-endpoint Chemprop
v2 models. Runs entirely on-device; no `admetica` pip package, no Flask, no extra
dependencies beyond what the ADMET-AI env already provides.

The 22 Admetica checkpoints ship inside the `admetica==1.4.1` sdist as Chemprop
v2.0 `.ckpt` files. `code/setup_admetica_models.sh` downloads them once, converts
each to Chemprop v2.1 (`.pt`) with the `chemprop` CLI, and writes the
applicability-domain mean vectors to `models/admetica/ad_vectors.json`. This client
loads those `.pt` files directly with chemprop 2.3.1 — nothing leaves the machine.

Usage:
    python3 admetica_client.py --smiles "CCO" "CC(=O)Oc1ccccc1C(=O)O" -o admetica.csv
    python3 admetica_client.py --input data.csv --smiles-column smiles -o admetica.csv
    python3 admetica_client.py --input mols.smi --properties Caco2,hERG -o admetica.csv
    python3 admetica_client.py --input data.csv --no-ad -o admetica.csv

Output: one row per molecule. Columns = input SMILES, then one column per requested
endpoint (default: all 22), then a paired `<endpoint>_AD` applicability-domain
column for each endpoint that has a mean vector (15 of 22). Classification endpoints
are probabilities in [0,1]; regression endpoints are predicted values in the
endpoint's native units. The AD score is the cosine similarity between the
molecule's Morgan fingerprint (r=2, 1024 bits) and the per-endpoint training-set mean
fingerprint — higher means more in-domain (range ~[0,1]).

Device: --device cpu (default) | mps | auto. These are single tiny MPNNs, so cpu is
recommended (mps dispatch overhead dominates the matmuls, as with ADMET-AI).

Note: upstream admetica's `include_probability` is effectively dead code — it
lowercases the model name before looking up the *capitalized* mean-vector keys, so
it always returns 0.0, and the CLI hard-codes it off. This client does the lookup
correctly, so the AD columns here are real.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Sequence

# --- Endpoint registry ------------------------------------------------------
# (canonical_name, model_stem, kind, ad_key_or_None). ad_key is the key into the
# Admetica mean_vectors dict (15 of 22 endpoints have one). Ordered by category.
ENDPOINTS: list[tuple[str, str, str, str | None]] = [
    # Absorption
    ("Caco2",          "caco2",            "regression",     "Caco2"),
    ("Lipophilicity",  "lipophilicity",    "regression",     "Lipophilicity"),
    ("Solubility",     "solubility",       "regression",     "Solubility"),
    ("Pgp-Inhibitor",  "pgp-inhibitor",    "classification",  None),
    ("Pgp-Substrate",  "pgp-substrate",    "classification",  "Pgp-Substrate"),
    # Distribution
    ("PPBR",           "ppbr",             "regression",     "PPBR"),
    ("VDss",           "vdss",             "regression",     "VDss"),
    # Metabolism
    ("CYP1A2-Inhibitor",  "cyp1a2-inhibitor",  "classification", "CYP1A2-Inhibitor"),
    ("CYP1A2-Substrate",  "cyp1a2-substrate",  "classification", None),
    ("CYP2C19-Inhibitor", "cyp2c19-inhibitor", "classification", "CYP2C19-Inhibitor"),
    ("CYP2C19-Substrate", "cyp2c19-substrate", "classification", None),
    ("CYP2C9-Inhibitor",  "cyp2c9-inhibitor",  "classification", "CYP2C9-Inhibitor"),
    ("CYP2C9-Substrate",  "cyp2c9-substrate",  "classification", "CYP2C9-Substrate"),
    ("CYP2D6-Inhibitor",  "cyp2d6-inhibitor",  "classification", "CYP2D6-Inhibitor"),
    ("CYP2D6-Substrate",  "cyp2d6-substrate",  "classification", "CYP2D6-Substrate"),
    ("CYP3A4-Inhibitor",  "cyp3a4-inhibitor",  "classification", None),
    ("CYP3A4-Substrate",  "cyp3a4-substrate",  "classification", None),
    # Excretion
    ("CL-Hepa",        "cl-hepa",          "regression",     "CL-Hepa"),
    ("CL-Micro",       "cl-micro",         "regression",     "CL-Micro"),
    ("Half-Life",      "half-life",        "regression",     "Half-Life"),
    # Toxicity
    ("hERG",           "herg",             "classification",  None),
    ("LD50",           "ld50",             "regression",      None),
]

BY_NAME = {name: (name, stem, kind, ad) for name, stem, kind, ad in ENDPOINTS}
ALL_NAMES = [name for name, *_ in ENDPOINTS]

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "admetica"


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
    if device == "auto":
        import torch
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return device


def filter_valid_smiles(smiles_list: Sequence[str]) -> tuple[list[str], list[tuple[int, str]]]:
    """Drop blank/invalid SMILES, returning (valid, [(orig_index, bad_smiles)])."""
    from rdkit import Chem
    valid: list[str] = []
    bad: list[tuple[int, str]] = []
    for i, s in enumerate(smiles_list):
        s = (s or "").strip()
        if not s:
            bad.append((i, "<blank>"))
            continue
        if Chem.MolFromSmiles(s) is None:
            bad.append((i, s))
            continue
        valid.append(s)
    return valid, bad


def parse_properties(spec: Sequence[str] | None) -> list[str]:
    """Resolve a --properties spec (comma-joined and/or repeated) to canonical names."""
    if not spec:
        return list(ALL_NAMES)
    names: list[str] = []
    for item in spec:
        for raw in item.split(","):
            raw = raw.strip()
            if not raw:
                continue
            key = raw
            # case-insensitive match against canonical names
            if key not in BY_NAME:
                ci = next((n for n in ALL_NAMES if n.lower() == key.lower()), None)
                if ci is None:
                    raise SystemExit(f"ERROR: unknown endpoint {raw!r}. "
                                     f"Available: {', '.join(ALL_NAMES)}")
                key = ci
            names.append(key)
    # de-dup preserving order
    seen: set[str] = set()
    out = [n for n in names if not (n in seen or seen.add(n))]
    return out


class AdmeticaPredictor:
    """Lazily loads and caches converted Admetica .pt models + AD mean vectors."""

    def __init__(self, models_dir: Path = MODELS_DIR, device: str = "cpu"):
        self.models_dir = Path(models_dir)
        self.device = device
        self._cache: dict[str, object] = {}
        self._ad_vectors: dict[str, list[float]] | None = None
        self._ad_loaded = False

    def _model_path(self, stem: str) -> Path:
        return self.models_dir / f"{stem}.pt"

    def _load_model(self, stem: str):
        path = self._model_path(stem)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `bash code/setup_admetica_models.sh` first."
            )
        from chemprop import models
        m = models.MPNN.load_from_checkpoint(str(path), map_location=self.device)
        if self.device == "mps":
            m.device = "mps"  # mirror admet_ai_client: force mps after init
        m.eval()
        return m

    def get_model(self, stem: str):
        if stem not in self._cache:
            self._cache[stem] = self._load_model(stem)
        return self._cache[stem]

    def ad_vectors(self) -> dict[str, list[float]]:
        if not self._ad_loaded:
            self._ad_loaded = True
            p = self.models_dir / "ad_vectors.json"
            if p.exists():
                with open(p) as f:
                    self._ad_vectors = json.load(f)
            else:
                self._ad_vectors = {}
        return self._ad_vectors or {}

    def predict_endpoint(self, stem: str, smiles: list[str]) -> list[float]:
        """Run one endpoint's MPNN over a batch of valid SMILES; returns one float each."""
        from chemprop import data, featurizers
        from lightning import pytorch as pl
        import torch
        m = self.get_model(stem)
        datapoints = [data.MoleculeDatapoint.from_smi(s) for s in smiles]
        dataset = data.MoleculeDataset(datapoints, featurizer=featurizers.SimpleMoleculeMolGraphFeaturizer())
        trainer = pl.Trainer(logger=False, enable_progress_bar=False,
                             accelerator=self.device, devices=1)
        with torch.no_grad(), contextlib.redirect_stderr(None), \
             warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence the benign "Dropping last batch of size 1" note
            loader = data.build_dataloader(dataset, shuffle=False)
            out = trainer.predict(m, loader)
        # flatten (batches, n, 1) -> list of floats, one per molecule
        flat: list[float] = []
        for batch in out:
            for row in batch:
                flat.append(float(row.item()) if row.ndim else float(row))
        return flat

    def ad_scores(self, ad_key: str, smiles: list[str]) -> list[float]:
        """Cosine similarity of each molecule's Morgan FP to the endpoint mean vector."""
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mean = self.ad_vectors().get(ad_key)
        if mean is None:
            return [float("nan")] * len(smiles)
        mean = np.asarray(mean, dtype=float)
        mean_norm = np.linalg.norm(mean)
        if mean_norm == 0:
            return [0.0] * len(smiles)
        scores: list[float] = []
        for s in smiles:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                scores.append(float("nan"))
                continue
            fp = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024),
                            dtype=float)
            fn = np.linalg.norm(fp)
            scores.append(float(np.dot(mean, fp) / (mean_norm * fn)) if fn else 0.0)
        return scores


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Predict ADMET locally with the Admetica Chemprop models.")
    p.add_argument("--input", help="path to .smi/.txt/.csv/.xlsx with SMILES")
    p.add_argument("--smiles-column", help="column name for SMILES in a .csv/.xlsx")
    p.add_argument("--smiles", nargs="*", help="one or more SMILES strings on the CLI")
    p.add_argument("--output", "-o", default="admetica_results.csv", help="output CSV path")
    p.add_argument("--properties", nargs="*",
                   help="endpoints to predict (comma-joined and/or repeated); default all 22")
    p.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu",
                   help="accelerator (default cpu; mps supported but slower for these tiny models)")
    p.add_argument("--no-ad", action="store_true",
                   help="drop the applicability-domain (<endpoint>_AD) columns")
    args = p.parse_args(argv)

    if args.input:
        smiles = load_smiles(args.input, args.smiles_column)
    elif args.smiles:
        smiles = list(args.smiles)
    else:
        p.error("provide --input or --smiles")
    if not smiles:
        p.error("no SMILES found in input")

    endpoints = parse_properties(args.properties)
    include_ad = not args.no_ad

    # Silence RDKit's MorganGenerator deprecation warnings (we keep the legacy
    # GetMorganFingerprintAsBitVect to match how Admetica's mean vectors were built).
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    device = resolve_device(args.device)
    valid, bad = filter_valid_smiles(smiles)
    for idx, s in bad:
        print(f"WARNING: dropping invalid SMILES at input row {idx}: {s!r}", file=sys.stderr)
    if not valid:
        print("ERROR: no valid SMILES to predict after filtering", file=sys.stderr)
        return 1

    print(f"Loading Admetica (device={device}); predicting {len(valid)}/{len(smiles)} "
          f"molecule(s) x {len(endpoints)} endpoint(s) ...", file=sys.stderr)

    pred = AdmeticaPredictor(device=device)

    # Build columns: smiles, then for each endpoint a value col (+ AD col when available).
    import pandas as pd
    cols: dict[str, list] = {"smiles": list(valid)}
    for name, stem, kind, ad in [BY_NAME[n] for n in endpoints]:
        try:
            vals = pred.predict_endpoint(stem, valid)
        except Exception as exc:  # noqa: BLE001 - surface a clean error, keep going for others
            print(f"ERROR: endpoint {name} failed: {exc}", file=sys.stderr)
            vals = [float("nan")] * len(valid)
        cols[name] = vals
        if include_ad and ad is not None:
            cols[f"{name}_AD"] = pred.ad_scores(ad, valid)

    out = pd.DataFrame(cols)
    out.to_csv(args.output, index=False)
    n_ad_cols = sum(1 for c in out.columns if c.endswith("_AD"))
    print(f"Wrote {len(out)} molecule(s) x {out.shape[1]} columns "
          f"({len(endpoints)} endpoints, {n_ad_cols} AD) -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())