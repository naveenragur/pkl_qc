#!/bin/bash
# Run pkl_qc with a Spack environment (same arguments as run_pkl_qc.sh).
#   ./run_pkl_qc_spack.sh all --config configs/gebco.json --out results_gebco --zones southamerica -j 8
#
# Setup follows /leonardo_work/DTGEO_T1_2/VMg_runPTHA/env_pygthm_source.source:
#   module load spack
#   spack env activate -p <spack env>
# Choose the environment with PKL_QC_SPACK_ENV=/path/to/env, or edit SPACK_ENV_DEFAULT below.
# It must provide python3 with numpy, pandas and matplotlib. To create one:
#   module load spack
#   spack env create -d /path/to/pkl_qc_env && spack env activate -p /path/to/pkl_qc_env
#   spack add python@3.11 py-numpy py-pandas py-matplotlib && spack concretize && spack install
cd "$(dirname "$(readlink -f "$0")")" || exit 1

SPACK_ENV_DEFAULT=/leonardo/home/userexternal/bbayrakt/pygthm_env/
SPACK_ENV_DIR=${PKL_QC_SPACK_ENV:-$SPACK_ENV_DEFAULT}

if [ -d "$SPACK_ENV_DIR" ] || [[ "$SPACK_ENV_DIR" != */* ]]; then :; else
    echo "pkl_qc: spack env '$SPACK_ENV_DIR' not found or not readable by $(id -un);" \
         "set PKL_QC_SPACK_ENV to an environment you can access" >&2
    exit 1
fi

module load spack || { echo "pkl_qc: 'module load spack' failed" >&2; exit 1; }
# site spack configs may print warnings and return non-zero even when activation worked;
# the checks below are the real test
spack env activate -p "$SPACK_ENV_DIR" 2>/dev/null \
    || echo "pkl_qc: warning: 'spack env activate $SPACK_ENV_DIR' returned non-zero" >&2
[ -n "${SPACK_ENV:-}" ] || { echo "pkl_qc: spack env '$SPACK_ENV_DIR' is not active" >&2; exit 1; }

# keep user/site packages from mixing into the spack environment
unset PYTHONPATH
export PYTHONNOUSERSITE=1

python3 -c "import numpy, pandas, matplotlib" || {
    echo "pkl_qc: spack env '$SPACK_ENV_DIR' ($(command -v python3)) lacks numpy/pandas/matplotlib" >&2; exit 1; }

export MPLCONFIGDIR="$PWD/.cache/mpl"
exec python3 -m pkl_qc "$@"
