#!/usr/bin/env python3
"""
admet_tool.py — ADMET prediction as an LLM-callable tool (ADMET-AI / Chemprop v2).

------------------------------------------------------------------------------
Install:  pip install admet-ai
  Pulls chemprop, torch, rdkit, pandas, numpy, lightning transitively (~0.5-1 GB).
  The ~13 MB of model weights ship inside the admet-ai wheel — no separate
  download, no network access at runtime. Python >= 3.11.
Run:      python3 admet_tool.py "CCO" "OC1=CC=C(C=C1)C=CC(=O)O"
------------------------------------------------------------------------------

Exposes `predict_admet`, suitable to register as an LLM tool (see
ADMET_TOOL_SCHEMA). The model is loaded once and cached for the process.
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")  # silence plain tqdm bars

_MODEL = None


def predict_admet(smiles, include_percentiles: bool = False) -> str:
    """Predict ADMET properties (absorption, distribution, metabolism, excretion,
    toxicity) for one or more molecules from SMILES, using a local Chemprop v2
    model (ADMET-AI). No network access.

    Use this to screen molecules for drug-likeness / safety / PK: BBB permeability,
    CYP inhibition/substrate, hERG, AMES mutagenicity, DILI, LD50, solubility,
    Caco-2, clearance, PAINS/BRENK/NIH alerts, and 40+ other endpoints.

    Args:
        smiles: a single SMILES string, or a list of them (batch).
        include_percentiles: if True, also include each endpoint's percentile rank
            vs. DrugBank approved drugs (0.9 = higher than 90% of approved drugs).

    Returns:
        A string, one block per molecule (input order): "<n> <smiles>" then
        "  name: value" lines for each endpoint. Invalid SMILES yield an error
        line instead of aborting the batch. Classification endpoints are
        probabilities in [0,1]; regression endpoints are predicted values.
    """
    global _MODEL
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")  # silence "SMILES Parse Error" on bad input

    single = isinstance(smiles, str)
    items = [smiles] if single else list(smiles)

    valid, idx = [], []
    blocks = [""] * len(items)
    for i, s in enumerate(items):
        s = (s or "").strip()
        if s and Chem.MolFromSmiles(s) is not None:
            valid.append(s)
            idx.append(i)
        else:
            blocks[i] = f"[{i + 1}] {s!r}\n  ERROR: invalid SMILES"

    if valid:
        if _MODEL is None:
            from admet_ai import ADMETModel
            _MODEL = ADMETModel()
            _MODEL.device = "cpu"  # CPU is faster than MPS for these tiny models

        # chemprop/lightning print tqdm bars + rank_zero chatter to stdout/stderr;
        # redirect both to devnull during predict (real errors still raise).
        devnull = os.open(os.devnull, os.O_WRONLY)
        so, se = os.dup(1), os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            preds = _MODEL.predict(smiles=valid)  # DataFrame aligned to `valid`
        finally:
            os.dup2(so, 1)
            os.dup2(se, 2)
            os.close(devnull)
            os.close(so)
            os.close(se)

        for j, i in enumerate(idx):
            row = preds.iloc[j]
            lines = [f"[{i + 1}] {valid[j]}"]
            for c in preds.columns:
                if not include_percentiles and c.endswith("_drugbank_approved_percentile"):
                    continue
                v = row[c]
                lines.append(f"  {c}: {float(v):.4g}" if hasattr(v, "__float__") else f"  {c}: {v}")
            blocks[i] = "\n".join(lines)

    return "\n\n".join(blocks)


ADMET_TOOL_SCHEMA = {
    "name": "predict_admet",
    "description": (
        "Predict ADMET properties (absorption, distribution, metabolism, excretion, "
        "toxicity) for small molecules from SMILES strings using a local Chemprop v2 "
        "model. Returns a readable text string with ~52 endpoints per molecule (BBB, "
        "CYP, hERG, AMES, DILI, LD50, solubility, Caco-2, clearance, PAINS/BRENK alerts, "
        "etc.). No network access. Pass one SMILES string or a list of them; invalid "
        "SMILES are reported per-molecule, not fatal."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "smiles": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "A single SMILES string, or a list of SMILES strings to predict in batch.",
            },
            "include_percentiles": {
                "type": "boolean",
                "description": "If true, also return each endpoint's percentile rank vs. DrugBank approved drugs.",
                "default": False,
            },
        },
        "required": ["smiles"],
    },
}


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    print(predict_admet(args if len(args) > 1 else (args[0] if args else ["CCO", "OC1=CC=C(C=C1)C=CC(=O)O"])))