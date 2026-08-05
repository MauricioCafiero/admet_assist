#!/usr/bin/env python3
"""
admetica_tool.py — Admetica ADMET prediction as an LLM-callable tool.

------------------------------------------------------------------------------
Install:  one-time `bash code/setup_admetica_models.sh` (fetches + converts the
  22 Admetica Chemprop checkpoints into models/admetica/*.pt). No `admetica` pip
  package and no extra deps — runs on the same chemprop/rdkit/torch the ADMET-AI
  client already uses.
Run:      python3 admetica_tool.py "CCO" "CC(=O)Oc1ccccc1C(=O)O"
------------------------------------------------------------------------------

Exposes `predict_admetica`, suitable to register as an LLM tool (see
ADMETICA_TOOL_SCHEMA). Models are loaded once and cached for the process.

Admetica covers 22 endpoints (5 absorption, 2 distribution, 10 CYP metabolism,
3 excretion, 2 toxicity). 15 of them ship an applicability-domain mean vector,
reported as a cosine-similarity score (higher = more in-domain, ~[0,1]). Note
this is a *different* model set from ADMET-AI (admet_tool.py) — useful as an
independent second opinion and the only source here for CYP1A2-substrate,
CYP2C19-substrate and Pgp-substrate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")  # silence plain tqdm bars

# scripts live in code/ — make admetica_client importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from admetica_client import (  # noqa: E402
    ALL_NAMES,
    BY_NAME,
    ENDPOINTS,
    AdmeticaPredictor,
    parse_properties,
)

_PRED = None


def predict_admetica(smiles, properties=None, include_ad: bool = True) -> str:
    """Predict ADMET properties for one or more molecules from SMILES, using the
    local Admetica Chemprop v2 models. No network access.

    Use this for an independent ADMET read on a molecule alongside ADMET-AI:
    Caco-2 / lipophilicity / solubility, PPBR, VDss, CYP1A2/2C9/2C19/2D6/3A4
    inhibition & substrate, Pgp inhibitor/substrate, hepatocyte/microsome
    clearance, half-life, hERG, LD50. The 15 endpoints that have a mean vector
    also report an applicability-domain cosine-similarity score.

    Args:
        smiles: a single SMILES string, or a list of them (batch).
        properties: optional list of endpoint names to limit to (case-insensitive,
            comma-joined items accepted). Default: all 22.
        include_ad: if True, append "(AD x.xx)" to each endpoint that has a
            training-set mean vector.

    Returns:
        A string, one block per molecule (input order): "[n] <smiles>" then
        "  Endpoint: value (AD x.xx)" lines. Invalid SMILES yield an error line
        instead of aborting the batch. Classification endpoints are probabilities
        in [0,1]; regression endpoints are predicted values.
    """
    global _PRED
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")  # silence "SMILES Parse Error" on bad input

    single = isinstance(smiles, str)
    items = [smiles] if single else list(smiles)
    endpoints = parse_properties(properties if properties is not None else None)
    registry = [BY_NAME[n] for n in endpoints]

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
        if _PRED is None:
            _PRED = AdmeticaPredictor(device="cpu")

        # chemprop/lightning print INFO + rank_zero chatter to stdout/stderr;
        # redirect both to devnull during prediction (real errors still raise).
        devnull = os.open(os.devnull, os.O_WRONLY)
        so, se = os.dup(1), os.dup(2)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            value_cols: dict[str, list[float]] = {}
            ad_cols: dict[str, list[float]] = {}
            for name, stem, _kind, ad_key in registry:
                value_cols[name] = _PRED.predict_endpoint(stem, valid)
                if include_ad and ad_key is not None:
                    ad_cols[name] = _PRED.ad_scores(ad_key, valid)
        finally:
            os.dup2(so, 1)
            os.dup2(se, 2)
            os.close(devnull)
            os.close(so)
            os.close(se)

        for j, i in enumerate(idx):
            lines = [f"[{i + 1}] {valid[j]}"]
            for name, _stem, kind, ad_key in registry:
                v = value_cols[name][j]
                vstr = f"{v:.4g}"
                tag = "prob" if kind == "classification" else "val"
                if include_ad and ad_key is not None:
                    lines.append(f"  {name}: {vstr} [{tag}, AD {ad_cols[name][j]:.3f}]")
                else:
                    lines.append(f"  {name}: {vstr} [{tag}]")
            blocks[i] = "\n".join(lines)

    return "\n\n".join(blocks)


ADMETICA_TOOL_SCHEMA = {
    "name": "predict_admetica",
    "description": (
        "Predict ADMET properties for small molecules from SMILES using the local "
        "Admetica Chemprop v2 models (22 endpoints: Caco-2, lipophilicity, solubility, "
        "PPBR, VDss, Pgp inhibitor/substrate, CYP1A2/2C9/2C19/2D6/3A4 inhibition & "
        "substrate, hepatocyte/microsome clearance, half-life, hERG, LD50). Independent "
        "of ADMET-AI, so useful as a second opinion. 15 endpoints also report an "
        "applicability-domain cosine-similarity score. No network access. Pass one "
        "SMILES string or a list; invalid SMILES are reported per-molecule, not fatal."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "smiles": {
                "type": ["string", "array"],
                "items": {"type": "string"},
                "description": "A single SMILES string, or a list of SMILES strings to predict in batch.",
            },
            "properties": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of endpoint names to limit to (case-insensitive). "
                    f"Default: all 22. Available: {', '.join(ALL_NAMES)}."
                ),
            },
            "include_ad": {
                "type": "boolean",
                "default": True,
                "description": "If true, append an applicability-domain cosine-similarity score per endpoint that has one.",
            },
        },
        "required": ["smiles"],
    },
}


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = ["CCO", "CC(=O)Oc1ccccc1C(=O)O"]
    print(predict_admetica(args if len(args) > 1 else args[0]))