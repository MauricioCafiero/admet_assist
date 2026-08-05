#!/usr/bin/env bash
# setup_admetica_models.sh — one-time fetch + convert of the Admetica model set.
#
# Admetica (https://github.com/datagrok-ai/admetica, MIT, (c) Datagrok / Oleksandra
# Serhiienko) ships 22 per-endpoint Chemprop v2.0 checkpoints inside its sdist. We
# don't install the `admetica` pip package (it hard-pins chemprop==2.0.0 /
# torch==2.4.0 / numpy==1.26.4, none of which have Python 3.14 wheels and which
# would conflict with the ADMET-AI env). Instead we pull just the checkpoints,
# convert each to Chemprop v2.1 once with the `chemprop` CLI already in this env,
# and load the converted .pt files directly with chemprop 2.3.1. No Flask, no
# second venv, no extra deps.
#
# Run with the project venv active (needs `chemprop` on PATH):
#     .venv/bin/python -m pip install -r requirements.txt   # if not already
#     . ./venv/bin/activate && bash code/setup_admetica_models.sh
#   or:
#     PATH="$PWD/.venv/bin:$PATH" bash code/setup_admetica_models.sh
#
# Output (gitignored): models/admetica/<endpoint>.pt  (22 files)
#                      models/admetica/ad_vectors.json (applicability-domain means)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/models/admetica"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

SDIST_URL="https://files.pythonhosted.org/packages/0f/b1/6ea5c3ccfbf2a2178e865aeb30649be800d411063ff97b62c63a978f3619/admetica-1.4.1.tar.gz"
SDIST_SHA256="e78ca5ff0be4d0716e4d93e53dd3663e92bc980f3c51f1f513fadf06397654d6"

CHEMPROP="$(command -v chemprop || true)"
if [ -z "$CHEMPROP" ]; then
    echo "ERROR: 'chemprop' CLI not found on PATH. Activate the project venv" >&2
    echo "       (e.g. . .venv/bin/activate) or set PATH=\$PWD/.venv/bin:\$PATH." >&2
    exit 1
fi

mkdir -p "$MODELS_DIR"
echo ">> downloading admetica-1.4.1 sdist (74 MB) ..."
SDIST="$WORK/admetica-1.4.1.tar.gz"
curl -fsSL "$SDIST_URL" -o "$SDIST"
echo "$SDIST_SHA256  $SDIST" | shasum -a 256 -c - >/dev/null

echo ">> extracting checkpoints + applicability-domain vectors ..."
tar -xzf "$SDIST" -C "$WORK"
SRC_MODELS="$WORK/admetica-1.4.1/admetica/Models"
SRC_CONST="$WORK/admetica-1.4.1/admetica/constants.py"
n_ckpts=$(find "$SRC_MODELS" -name '*.ckpt' | wc -l | tr -d ' ')
if [ "$n_ckpts" -ne 22 ]; then
    echo "ERROR: expected 22 .ckpt files, found $n_ckpts" >&2; exit 1
fi

echo ">> converting 22 checkpoints Chemprop v2.0 -> v2.1 (.pt) into $MODELS_DIR ..."
converted=0
for ckpt in "$SRC_MODELS"/*.ckpt; do
    stem="$(basename "$ckpt" .ckpt)"
    "$CHEMPROP" convert --conversion v2_0_to_v2_1 \
        -i "$ckpt" -o "$MODELS_DIR/$stem.pt" >/dev/null 2>&1
    converted=$((converted + 1))
    printf "   %-22s -> %s.pt\n" "$stem" "$stem"
done

echo ">> writing applicability-domain mean vectors -> $MODELS_DIR/ad_vectors.json ..."
PYTHON="$(dirname "$CHEMPROP")/python"
"$PYTHON" - "$SRC_CONST" "$MODELS_DIR/ad_vectors.json" <<'PY'
import json, sys, os, sys
# constants.py imports nothing heavy at module top (just a dict literal).
sys.path.insert(0, os.path.dirname(sys.argv[1]))
from constants import mean_vectors  # noqa: E402
with open(sys.argv[2], "w") as f:
    json.dump({k: list(v) for k, v in mean_vectors.items()}, f)
print(f"   {len(mean_vectors)} endpoint mean vectors written")
PY

echo ""
echo "Done. $converted models in $MODELS_DIR (gitignored)."
echo "Next: .venv/bin/python code/admetica_client.py --smiles 'CCO' -o out.csv"