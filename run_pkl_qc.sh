#!/bin/bash
# Run pkl_qc interactively (login node / workstation).
#   ./run_pkl_qc.sh all      --config configs/gebco.json --zones southamerica -j 8 --out results_gebco
#   ./run_pkl_qc.sh evaluate --config configs/gebco.json --out results_gebco
#   ./run_pkl_qc.sh plot     --config configs/gebco.json --out results_gebco --zones sunda2
# PKL_QC_PYTHON selects the interpreter (default: Leonardo venv if present, else python3).
cd "$(dirname "$(readlink -f "$0")")" || exit 1
LEONARDO_PY=/leonardo/home/userexternal/nraguram/venv_leonardo_py311/bin/python3.11
if [ -n "${PKL_QC_PYTHON:-}" ]; then PY=$PKL_QC_PYTHON
elif [ -x "$LEONARDO_PY" ]; then PY=$LEONARDO_PY
else PY=python3; fi
export MPLCONFIGDIR="$PWD/.cache/mpl"
exec "$PY" -m pkl_qc "$@"
