# admet_ai_client

Predict ADMET properties for molecules from SMILES **locally** using
[ADMET-AI](https://github.com/swansonk14/admet_ai) — a Chemprop v2 message-passing
neural-network ensemble. No network call, no flaky server; the 13 MB of model
weights ship inside the `admet-ai` pip wheel.

This replaced the earlier `admetlab3_client.py` (which hit the ADMETlab 3.0 web
service — its REST API was broken server-side and the site has since been
returning 502). That file is kept for reference but is not the active path.

## Repo layout

```
admet/
├── code/                       # Python entry points (tracked)
│   ├── admet_ai_client.py      # ADMET-AI batch CLI -> CSV (primary, 52 endpoints)
│   ├── admet_tool.py           # ADMET-AI LLM-callable tool -> string
│   ├── admetica_client.py      # Admetica batch CLI -> CSV (22 endpoints, independent 2nd opinion)
│   ├── admetica_tool.py        # Admetica LLM-callable tool -> string
│   ├── setup_admetica_models.sh# one-time fetch + convert of the Admetica checkpoints
│   ├── admetlab3_client.py     # legacy server client (abandoned, kept for reference)
│   └── make_pdf.py             # markdown -> PDF (run with the moltui mt-env python, which has reportlab)
├── models/admetica/            # converted Admetica .pt + ad_vectors.json (gitignored, ~74 MB)
├── requirements.txt
├── README.md
├── anna_top_15.csv             # input compound sheet (gitignored — local only)
└── anna_admet_report.{md,pdf}  # generated ADMET report (gitignored — local only)
```

`.venv/`, `__pycache__/`, `models/`, `anna_top_15.csv` and the generated
`anna_admet_report.*` are in `.gitignore` (Anna's compound data, the derived report,
and the fetched model weights stay local, not committed).

## Download & install

No separate model download is needed — the ~13 MB of Chemprop v2 weights ship
**inside the `admet-ai` pip wheel**, so installing the package is the only step.

```sh
# 1. create a virtualenv (Python >= 3.11)
python3 -m venv .venv

# 2. install dependencies (~0.5-1 GB: torch, rdkit, chemprop, pandas, numpy, lightning)
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` is just `admet-ai>=2.0.1` + `openpyxl>=3.1`; you can also install
directly with `.venv/bin/pip install admet-ai`. Tested on Python 3.14.5 / Apple
Silicon (admet-ai 2.0.1, torch 2.13.0, chemprop 2.3.1, rdkit 2026.3.4).

## Run

Two entry points: a batch **CLI client** (`admet_ai_client.py`, writes CSV) and an
**LLM tool** (`admet_tool.py`, returns a text string for an agent to read).

### CLI client — `admet_ai_client.py`

```sh
# SMILES on the command line
.venv/bin/python code/admet_ai_client.py --smiles "CC(=O)Oc1ccccc1C(=O)O" CCO c1ccccc1 -o admet.csv

# from a .smi/.txt file (one SMILES per line; leading '#' lines are comments)
.venv/bin/python code/admet_ai_client.py --input molecules.smi -o admet.csv

# from a .csv with a SMILES column
.venv/bin/python code/admet_ai_client.py --input data.csv --smiles-column smiles -o admet.csv

# drop the DrugBank percentile columns (half the width)
.venv/bin/python code/admet_ai_client.py --input data.csv --no-percentiles -o admet.csv

# exclude the RDKit-computed physicochemical properties (MW, logP, TPSA, alerts, ...)
.venv/bin/python code/admet_ai_client.py --input data.csv --no-physchem -o admet.csv
```

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `--input` | — | `.smi`/`.txt`/`.csv`/`.xlsx` of SMILES |
| `--smiles-column` | auto (first `smiles`-like col) | column name in a `.csv`/`.xlsx` |
| `--smiles` | — | one or more SMILES on the CLI |
| `--output` / `-o` | `admet_results.csv` | output CSV path |
| `--device` | `cpu` | `cpu` \| `mps` \| `auto` (see Device below) |
| `--no-percentiles` | off | drop the 52 `*_drugbank_approved_percentile` columns |
| `--no-physchem` | off | drop the RDKit-computed physicochemical endpoints |

Invalid SMILES are dropped with a `WARNING` on stderr (RDKit-prefiltered so one
bad molecule doesn't crash the batch); the remaining molecules are predicted and
written.

### LLM tool — `admet_tool.py`

A single function, `predict_admet(smiles, include_percentiles=False)`, designed to
be registered as a tool an LLM can call. It takes one SMILES string or a list
(batch) and returns a **string** (one block per molecule, `name: value` lines) —
compact and readable so an agent can interpret the result directly. Invalid
SMILES produce a per-molecule error line, never fatal. A ready-to-wire tool
definition is exported as `ADMET_TOOL_SCHEMA` (OpenAI/Anthropic function-calling
shape). All chemprop/lightning/RDKit stderr/stdout chatter is suppressed.

```sh
# run directly (positional args are SMILES)
.venv/bin/python code/admet_tool.py "CCO" "OC1=CC=C(C=C1)C=CC(=O)O"
```

```python
# use as a tool in Python / an agent harness
import sys; sys.path.insert(0, "code")  # scripts live in code/
from admet_tool import predict_admet, ADMET_TOOL_SCHEMA
print(predict_admet(["CCO", "OC1=CC=C(C=C1)C=CC(=O)O"], include_percentiles=True))
```

## Output

One row per molecule. By default **105 columns**: the input `smiles`, then **52
ADMET endpoints**, then **52 `_drugbank_approved_percentile`** columns (each
endpoint's value ranked against the DrugBank approved reference set — useful
context that ADMETlab doesn't provide). With `--no-percentiles` this drops to 53;
with `--no-physchem` the 11 RDKit-computed physchem endpoints (and their
percentiles) are removed.

The 52 endpoints (Physicochemical → Absorption → Distribution → Excretion →
Metabolism → Toxicity):

- **Physicochemical** (RDKit-computed): `molecular_weight`, `logP`,
  `hydrogen_bond_acceptors`, `hydrogen_bond_donors`, `Lipinski`, `QED`,
  `stereo_centers`, `tpsa`, `PAINS_alert`, `BRENK_alert`, `NIH_alert`
- **Absorption**: `HIA_Hou`, `Bioavailability_Ma`, `Solubility_AqSolDB`,
  `Lipophilicity_AstraZeneca`, `HydrationFreeEnergy_FreeSolv`, `Caco2_Wang`,
  `PAMPA_NCATS`, `Pgp_Broccatelli`
- **Distribution**: `BBB_Martins`, `PPBR_AZ`, `VDss_Lombardo`
- **Excretion**: `Half_Life_Obach`, `Clearance_Hepatocyte_AZ`,
  `Clearance_Microsome_AZ`
- **Metabolism**: `CYP1A2_Veith`, `CYP2C19_Veith`, `CYP2C9_Veith`, `CYP2D6_Veith`,
  `CYP3A4_Veith`, `CYP2C9_Substrate_CarbonMangels`, `CYP2D6_Substrate_CarbonMangels`,
  `CYP3A4_Substrate_CarbonMangels`
- **Toxicity**: `hERG`, `ClinTox`, `AMES`, `DILI`, `Carcinogens_Lagunin`,
  `LD50_Zhu`, `Skin_Reaction`, `NR-AR`, `NR-AR-LBD`, `NR-AhR`, `NR-Aromatase`,
  `NR-ER`, `NR-ER-LBD`, `NR-PPAR-gamma`, `SR-ARE`, `SR-ATAD5`, `SR-HSE`,
  `SR-MMP`, `SR-p53`

Classification endpoints (31) are probabilities in [0, 1]; regression endpoints
(21) are predicted values in the endpoint's native units. The percentile columns
are fractions in [0, 1] (e.g. 0.9 = higher than 90% of approved drugs).

## Device

`--device` controls the Lightning accelerator passed to Chemprop:

- **`cpu` (default, recommended).** Benchmarked at **~3.5 ms/mol** (n=1000) on
  this Apple Silicon machine.
- **`mps`.** Supported and produces identical results, but **~4.5 ms/mol** —
  slower than CPU. The Chemprop MPNNs are tiny (~1.3 MB each), so MPS
  dispatch/transfer overhead dominates the actual matmuls. `ADMETModel` only
  auto-selects CUDA (not MPS), so the CLI sets `model.device = "mps"` after init.
- **`auto`.** picks `mps` if `torch.backends.mps.is_available()` else `cpu`.

A single-molecule run prints a benign
`Dropping last batch of size 1 to avoid issues with batch normalization` warning
from Chemprop — predictions are still produced correctly.

## How it works

`ADMETModel` (from `admet_ai`) loads two multi-task Chemprop v2 ensembles bundled
in the wheel — `admet_classification/` (5 checkpoints) and `admet_regression/`
(5 checkpoints), ~13 MB total — and runs them via `lightning.Trainer.predict`.
Physicochemical properties and DrugBank percentiles are computed by ADMET-AI from
RDKit and a bundled `drugbank_approved.csv` reference set. Nothing leaves the
machine.

## Versions tested

admet-ai 2.0.1 · chemprop 2.3.1 · torch 2.13.0 · rdkit 2026.3.4 · Python 3.14.5,
macOS arm64.

---

# admetica_client (independent second opinion)

A second, **independent** ADMET predictor built on [Admetica](https://github.com/datagrok-ai/admetica)
(Datagrok, MIT) — 22 per-endpoint Chemprop v2 models trained on different datasets than
ADMET-AI. Useful as a cross-check on shared endpoints and the **only** source here for
CYP1A2-substrate, CYP2C19-substrate and Pgp-substrate (ADMET-AI doesn't model those).
Runs in the **same `.venv`** as ADMET-AI — no extra dependencies, no second venv.

We do **not** `pip install admetica`: that package hard-pins `chemprop==2.0.0`,
`torch==2.4.0`, `numpy==1.26.4` (none have Python 3.14 wheels) and drags in Flask for a
web server we don't use. Instead we pull just the 22 checkpoints from the
`admetica==1.4.1` sdist, convert each once to Chemprop v2.1, and load the `.pt` files
directly with chemprop 2.3.1.

## One-time model setup

```sh
PATH="$PWD/.venv/bin:$PATH" bash code/setup_admetica_models.sh
```

Downloads the sdist (74 MB, sha256-pinned), extracts the 22 `.ckpt` files, converts each
to `.pt` with `chemprop convert --conversion v2_0_to_v2_1`, and writes the
applicability-domain mean vectors to `models/admetica/ad_vectors.json`. Output is
gitignored under `models/admetica/`. Requires the `chemprop` CLI on PATH (from the venv
you already installed for ADMET-AI).

## Run

Same two entry-point shape as ADMET-AI: a batch **CLI client** (`admetica_client.py`,
writes CSV) and an **LLM tool** (`admetica_tool.py`, returns a text string).

### CLI client — `admetica_client.py`

```sh
# SMILES on the command line (all 22 endpoints + AD scores)
.venv/bin/python code/admetica_client.py --smiles "CC(=O)Oc1ccccc1C(=O)O" CCO -o admetica.csv

# from a .csv with a SMILES column
.venv/bin/python code/admetica_client.py --input data.csv --smiles-column smiles -o admetica.csv

# only some endpoints (comma-joined or repeated; case-insensitive)
.venv/bin/python code/admetica_client.py --input data.csv --properties Caco2,hERG,LD50 -o admetica.csv

# drop the applicability-domain columns
.venv/bin/python code/admetica_client.py --input data.csv --no-ad -o admetica.csv
```

Flags:

| Flag | Default | Purpose |
|---|---|---|
| `--input` | — | `.smi`/`.txt`/`.csv`/`.xlsx` of SMILES |
| `--smiles-column` | auto (first `smiles`-like col) | column name in a `.csv`/`.xlsx` |
| `--smiles` | — | one or more SMILES on the CLI |
| `--output` / `-o` | `admetica_results.csv` | output CSV path |
| `--properties` | all 22 | endpoints to predict (comma-joined and/or repeated) |
| `--device` | `cpu` | `cpu` \| `mps` \| `auto` (cpu recommended; these are single tiny MPNNs) |
| `--no-ad` | off | drop the `<endpoint>_AD` applicability-domain columns |

Invalid SMILES are dropped with a `WARNING` on stderr (RDKit-prefiltered); the rest are
predicted and written.

### LLM tool — `admetica_tool.py`

`predict_admetica(smiles, properties=None, include_ad=True)` — one SMILES string or a
list (batch), returns a **string** (one block per molecule, `Endpoint: value [val|prob,
AD x.xxx]` lines). Invalid SMILES produce a per-molecule error line, never fatal. A
ready-to-wire tool definition is exported as `ADMETICA_TOOL_SCHEMA`.

```sh
.venv/bin/python code/admetica_tool.py "CCO" "CC(=O)Oc1ccccc1C(=O)O"
```

```python
import sys; sys.path.insert(0, "code")
from admetica_tool import predict_admetica, ADMETICA_TOOL_SCHEMA
print(predict_admetica(["CCO", "CC(=O)Oc1ccccc1C(=O)O"], include_ad=True))
```

## Output

One row per molecule. By default **38 columns**: input `smiles`, then 22 endpoint value
columns, then 15 `<endpoint>_AD` columns (one per endpoint that has a training-set mean
vector). With `--no-ad` this drops to 23; selecting a `--properties` subset narrows
accordingly.

The 22 endpoints (Absorption → Distribution → Metabolism → Excretion → Toxicity):

- **Absorption**: `Caco2`, `Lipophilicity`, `Solubility` (regression), `Pgp-Inhibitor`,
  `Pgp-Substrate` (classification)
- **Distribution**: `PPBR`, `VDss` (regression)
- **Metabolism**: `CYP1A2/2C19/2C9/2D6/3A4-Inhibitor` & `-Substrate` (10, classification)
- **Excretion**: `CL-Hepa`, `CL-Micro`, `Half-Life` (regression)
- **Toxicity**: `hERG` (classification), `LD50` (regression)

Classification endpoints (13) are probabilities in [0, 1]; regression endpoints (9) are
predicted values in the endpoint's native units. The **AD** column is the cosine
similarity between the molecule's Morgan fingerprint (r=2, 1024 bits) and the endpoint's
training-set mean fingerprint — higher means more in-domain (~[0,1]). Treat a prediction
as low-confidence when its AD score is well below the bulk (e.g. ethanol's `PPBR` comes
out -258.5 with AD 0.30 — clearly out of domain).

> **Note on the AD score:** upstream Admetica's `include_probability` is effectively dead
> code — it lowercases the model name before looking up the *capitalized* mean-vector
> keys, so it always returns 0.0, and the CLI hard-codes it off. This client does the
> lookup correctly, so the `_AD` columns here are real.

## How it works

`AdmeticaPredictor` loads each converted `.pt` with `chemprop.models.MPNN.load_from_checkpoint`
and runs it via `lightning.Trainer.predict`, one endpoint at a time. AD scores are
computed with RDKit Morgan fingerprints + a per-endpoint mean vector. Nothing leaves the
machine.

## Versions tested

admetica 1.4.1 (model checkpoints only; package not installed) · chemprop 2.3.1 ·
torch 2.13.0 · rdkit 2026.3.4 · Python 3.14.5, macOS arm64.