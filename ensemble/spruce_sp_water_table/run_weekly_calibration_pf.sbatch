#!/bin/bash
#SBATCH --job-name=SPWTweekly
#SBATCH --account=hpcl-cli185
#SBATCH --partition=parallel
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/hpcl-cli185/zdr/e3sm_run/UQ/analysis/SPRSPWT512_20261028_weekly/weekly_%j.log

set -euo pipefail

OLMT_ROOT=/projects/hpcl-cli185/proj-shared/zdr/elm-olmt
ANALYSIS_ROOT=/scratch/hpcl-cli185/zdr/e3sm_run/UQ/analysis/SPRSPWT512_20261028_weekly
ENSEMBLE_ROOT=/scratch/hpcl-cli185/zdr/e3sm_run/UQ/ensembles/SPRSPWT512_20261028_US-SPR_ICBELMBC_TAMB
PYTHON=/projects/hpcl-cli185/proj-shared/zdr/conda/envs/OLMT_pf/bin/python

export PYTHONPATH="$OLMT_ROOT"
export OLMT_SURROGATE_WORKERS=8
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$ANALYSIS_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$OLMT_ROOT"

"$PYTHON" ensemble/spruce_sp_water_table/calibrate_weekly_wt.py "$1" \
  --ensemble-root "$ENSEMBLE_ROOT" \
  --output "$ANALYSIS_ROOT" \
  --workers 8 \
  "${@:2}"
