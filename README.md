# admet

Local ADMET prediction for molecules from SMILES. Two **independent**
Chemprop v2 message-passing neural-network model families run on the same
machine, in the same virtualenv, with no network calls:

- **ADMET-AI** — a multi-task MPNN ensemble, 52 endpoints, weights bundled in the
  `admet-ai` pip wheel (~13 MB). Adds DrugBank approved **percentiles**.
- **Admetica** — 22 per-endpoint MPNNs trained on different datasets, used as an
  independent second opinion. Adds per-endpoint **applicability-domain** scores.

Each model ships as a batch **CLI client** (writes CSV) and an **LLM-callable
tool** (returns a text string for an agent to read) — see [LLM tools](#llm-tools).

## Repo layout

```
admet/
├── code/
│   ├── admet_ai_client.py      # ADMET-AI batch CLI -> CSV (52 endpoints)
│   ├── admet_tool.py           # ADMET-AI LLM-callable tool -> string
│   ├── admetica_client.py      # Admetica batch CLI -> CSV (22 endpoints)
│   ├── admetica_tool.py        # Admetica LLM-callable tool -> string
│   ├── setup_admetica_models.sh# one-time fetch + convert of the Admetica checkpoints
│   ├── make_pdf.py             # markdown -> PDF (run with the moltui mt-env python, which has reportlab)
│   └── admetlab3_client.py     # legacy server client (kept for reference, not used)
├── models/admetica/            # converted Admetica .pt + ad_vectors.json (gitignored, ~74 MB)
├── requirements.txt
└── README.md
```

`models/`, `anna_top_15.csv`, and the generated `anna_admet_report.*` /
`anna_admetica.*` files are gitignored (compound data, derived reports, and
fetched model weights stay local).

## Install

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt   # ~0.5-1 GB: torch, rdkit, chemprop, pandas, numpy, lightning
```

`requirements.txt` is `admet-ai>=2.0.1` + `openpyxl>=3.1`. ADMET-AI needs no
separate model download — the weights ship inside the wheel. Admetica needs a
one-time checkpoint fetch (see [Admetica setup](#admetica-setup)). Tested on
Python 3.14.5 / Apple Silicon (admet-ai 2.0.1, torch 2.13.0, chemprop 2.3.1,
rdkit 2026.3.4).

## Models, side by side

| | ADMET-AI | Admetica |
|---|---|---|
| Endpoints | 52 | 22 |
| Architecture | 2 multi-task ensembles (5 ckpt each) | 22 single per-endpoint MPNNs |
| Training data | ADMET-AI datasets | different datasets (Datagrok) |
| Extra context | DrugBank approved **percentiles** (52 cols) | per-endpoint **applicability-domain** scores |
| Install | `pip install admet-ai` (weights in wheel) | checkpoints pulled from sdist + converted |
| CLI client | `code/admet_ai_client.py` | `code/admetica_client.py` |
| LLM tool | `code/admet_tool.py` | `code/admetica_tool.py` |
| Default output width | 105 cols | 38 cols |

**Unique to ADMET-AI:** PAINS/BRENK/NIH alerts, PAMPA, HIA, Bioavailability, BBB,
ClinTox, AMES, DILI, the NR-/SR- Tox21 panels, and DrugBank percentiles.
**Unique to Admetica:** Pgp-Substrate, CYP1A2/2C19-substrate, and applicability-
domain confidence scores.

**Shared endpoints** (different models — useful to cross-check): `Caco2`,
`Lipophilicity`/`logP`-like, `Solubility`, `Pgp` (inhibition), `PPBR`, `VDss`,
`Half-Life`, microsome/hepatocyte clearance, `CYP1A2/2C19/2C9/2D6/3A4` inhibition,
`hERG`, `LD50`. Naming differs (e.g. ADMET-AI `Caco2_Wang` vs Admetica `Caco2`)
and units/thresholds are model-specific — compare trends, not raw numbers.

### CLI usage

Both clients take SMILES from the command line, a `.smi`/`.txt` file (one SMILES
per line; leading `#` lines are comments), or a `.csv`/`.xlsx` with a SMILES
column; both write one row per molecule and drop invalid SMILES with a
`WARNING` on stderr (RDKit-prefiltered so one bad molecule doesn't crash the
batch).

```sh
# ADMET-AI — 52 endpoints + DrugBank percentiles (105 cols by default)
.venv/bin/python code/admet_ai_client.py --smiles "CC(=O)Oc1ccccc1C(=O)O" CCO -o admet.csv
.venv/bin/python code/admet_ai_client.py --input data.csv --smiles-column smiles -o admet.csv
.venv/bin/python code/admet_ai_client.py --input data.csv --no-percentiles -o admet.csv   # -> 53 cols
.venv/bin/python code/admet_ai_client.py --input data.csv --no-physchem     -o admet.csv   # drop RDKit physchem

# Admetica — 22 endpoints + AD scores (38 cols by default)
.venv/bin/python code/admetica_client.py --smiles "CC(=O)Oc1ccccc1C(=O)O" CCO -o admetica.csv
.venv/bin/python code/admetica_client.py --input data.csv --smiles-column smiles -o admetica.csv
.venv/bin/python code/admetica_client.py --input data.csv --properties Caco2,hERG,LD50 -o admetica.csv
.venv/bin/python code/admetica_client.py --input data.csv --no-ad -o admetica.csv          # -> 23 cols
```

### CLI flags

| Flag | ADMET-AI | Admetica |
|---|---|---|
| `--input` | `.smi`/`.txt`/`.csv`/`.xlsx` of SMILES | same |
| `--smiles-column` | auto (first `smiles`-like col) | same |
| `--smiles` | one or more SMILES on the CLI | same |
| `--output` / `-o` | `admet_results.csv` | `admetica_results.csv` |
| `--device` | `cpu` \| `mps` \| `auto` (default `cpu`) | same (cpu recommended) |
| `--no-percentiles` | drop 52 `*_drugbank_approved_percentile` cols | n/a |
| `--no-physchem` | drop RDKit physicochemical endpoints | n/a |
| `--properties` | n/a | endpoints to predict (comma-joined/repeated) |
| `--no-ad` | n/a | drop `<endpoint>_AD` applicability-domain cols |

### Endpoints

**ADMET-AI — 52 endpoints** (Physicochemical → Absorption → Distribution →
Excretion → Metabolism → Toxicity). Classification endpoints (31) are
probabilities in [0, 1]; regression endpoints (21) are predicted values in
native units. Percentile columns are fractions in [0, 1] (0.9 = higher than 90%
of approved drugs).

- **Physicochemical** (RDKit-computed): `molecular_weight`, `logP`,
  `hydrogen_bond_acceptors`, `hydrogen_bond_donors`, `Lipinski`, `QED`,
  `stereo_centers`, `tpsa`, `PAINS_alert`, `BRENK_alert`, `NIH_alert`
- **Absorption**: `HIA_Hou`, `Bioavailability_Ma`, `Solubility_AqSolDB`,
  `Lipophilicity_AstraZeneca`, `HydrationFreeEnergy_FreeSolv`, `Caco2_Wang`,
  `PAMPA_NCATS`, `Pgp_Broccatelli`
- **Distribution**: `BBB_Martins`, `PPBR_AZ`, `VDss_Lombardo`
- **Excretion**: `Half_Life_Obach`, `Clearance_Hepatocyte_AZ`,
  `Clearance_Microsome_AZ`
- **Metabolism**: `CYP1A2_Veith`, `CYP2C19_Veith`, `CYP2C9_Veith`,
  `CYP2D6_Veith`, `CYP3A4_Veith`, `CYP2C9_Substrate_CarbonMangels`,
  `CYP2D6_Substrate_CarbonMangels`, `CYP3A4_Substrate_CarbonMangels`
- **Toxicity**: `hERG`, `ClinTox`, `AMES`, `DILI`, `Carcinogens_Lagunin`,
  `LD50_Zhu`, `Skin_Reaction`, `NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`,
  `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`, `SR-ARE`, `SR-ATAD5`, `SR-HSE`,
  `SR-MMP`, `SR-p53`

**Admetica — 22 endpoints** (Absorption → Distribution → Metabolism → Excretion
→ Toxicity). Classification endpoints (13) are probabilities in [0, 1];
regression endpoints (9) are predicted values in native units. The **AD** column
is the cosine similarity between the molecule's Morgan fingerprint (r=2, 1024
bits) and the endpoint's training-set mean fingerprint — higher means more
in-domain (~[0,1]); treat a prediction as low-confidence when its AD score is
well below the bulk (e.g. ethanol's `PPBR` comes out -258.5 with AD 0.30 —
clearly out of domain).

- **Absorption**: `Caco2`, `Lipophilicity`, `Solubility` (regression),
  `Pgp-Inhibitor`, `Pgp-Substrate` (classification)
- **Distribution**: `PPBR`, `VDss` (regression)
- **Metabolism**: `CYP1A2/2C19/2C9/2D6/3A4-Inhibitor` & `-Substrate`
  (10, classification)
- **Excretion**: `CL-Hepa`, `CL-Micro`, `Half-Life` (regression)
- **Toxicity**: `hERG` (classification), `LD50` (regression)

## Admetica setup

Admetica's `admetica` package is **not** pip-installed (it hard-pins
`chemprop==2.0.0`, `torch==2.4.0`, `numpy==1.26.4` — no Python 3.14 wheels — and
drags in an unused Flask web server). Only its 22 checkpoints are pulled from the
`admetica==1.4.1` sdist, converted once to Chemprop v2.1, and loaded as `.pt`
with chemprop 2.3.1.

```sh
PATH="$PWD/.venv/bin:$PATH" bash code/setup_admetica_models.sh
```

Downloads the sdist (74 MB, sha256-pinned), extracts the 22 `.ckpt` files,
converts each to `.pt` with `chemprop convert --conversion v2_0_to_v2_1`, and
writes per-endpoint applicability-domain mean vectors to
`models/admetica/ad_vectors.json`. Output is gitignored under `models/admetica/`.
Requires the `chemprop` CLI on PATH (from the ADMET-AI venv).

## LLM tools

Each model also ships as an LLM-callable tool — a single function that takes one
SMILES string or a list (batch) and returns a compact text string (one block per
molecule) for an agent to interpret directly. Invalid SMILES produce a
per-molecule error line, never fatal; all chemprop/lightning/RDKit chatter is
suppressed. Each exports a ready-to-wire `*_TOOL_SCHEMA` (OpenAI/Anthropic
function-calling shape).

| | `admet_tool.py` | `admetica_tool.py` |
|---|---|---|
| Function | `predict_admet(smiles, include_percentiles=False)` | `predict_admetica(smiles, properties=None, include_ad=True)` |
| Schema | `ADMET_TOOL_SCHEMA` | `ADMETICA_TOOL_SCHEMA` |
| Per-endpoint line | `name: value` | `Endpoint: value [val\|prob, AD x.xxx]` |
| Endpoint subset | no (always 52) | yes (`properties=[...]`) |
| Extra toggle | `include_percentiles` | `include_ad` |

Run directly (positional args are SMILES):

```sh
.venv/bin/python code/admet_tool.py    "CCO" "OC1=CC=C(C=C1)C=CC(=O)O"
.venv/bin/python code/admetica_tool.py "CCO" "CC(=O)Oc1ccccc1C(=O)O"
```

Use as a tool in Python / an agent harness:

```python
import sys; sys.path.insert(0, "code")
from admet_tool import predict_admet, ADMET_TOOL_SCHEMA
print(predict_admet(["CCO", "OC1=CC=C(C=C1)C=CC(=O)O"], include_percentiles=True))
```
```python
import sys; sys.path.insert(0, "code")
from admetica_tool import predict_admetica, ADMETICA_TOOL_SCHEMA
print(predict_admetica(["CCO", "CC(=O)Oc1ccccc1C(=O)O"], include_ad=True))
```

## Device

`--device` controls the Lightning accelerator passed to Chemprop:

- **`cpu` (default, recommended).** ~3.5 ms/mol (n=1000) on this Apple Silicon
  machine for ADMET-AI.
- **`mps`.** Supported and produces identical results, but slower (~4.5 ms/mol)
  — the Chemprop MPNNs are tiny (~1.3 MB each), so MPS dispatch/transfer overhead
  dominates the matmuls. `ADMETModel` only auto-selects CUDA (not MPS), so the
  ADMET-AI CLI sets `model.device = "mps"` after init. Admetica's single tiny
  MPNNs see no benefit from MPS either.
- **`auto`.** picks `mps` if `torch.backends.mps.is_available()` else `cpu`.

A single-molecule ADMET-AI run prints a benign
`Dropping last batch of size 1 to avoid issues with batch normalization` warning
from Chemprop — predictions are still produced correctly.

## How it works

- **ADMET-AI:** `ADMETModel` loads the `admet_classification/` and
  `admet_regression/` ensembles bundled in the wheel and runs them via
  `lightning.Trainer.predict`. Physicochemical properties and DrugBank
  percentiles are computed by ADMET-AI from RDKit and a bundled
  `drugbank_approved.csv` reference set.
- **Admetica:** `AdmeticaPredictor` loads each converted `.pt` with
  `chemprop.models.MPNN.load_from_checkpoint` and runs it via
  `lightning.Trainer.predict`, one endpoint at a time. AD scores use RDKit
  Morgan fingerprints + a per-endpoint mean vector.

Nothing leaves the machine.

## Versions tested

admet-ai 2.0.1 · chemprop 2.3.1 · torch 2.13.0 · rdkit 2026.3.4 · Python 3.14.5,
macOS arm64. Admetica 1.4.1 checkpoints (package not installed).