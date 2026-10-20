#!/usr/bin/env bash
#SBATCH --job-name=build-images
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --partition=CPU-amd
##SBATCH --account=<your-account>
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_ROOT/scripts/slurm/common.sh"

# Apptainer converts large Docker layers using temp space; the default /tmp may
# be a small RAM-backed tmpfs, which triggers OOM. Use node-local disk scratch.
SCRATCH="/scratch/${USER}_${SLURM_JOB_ID}"
export APPTAINER_TMPDIR="$SCRATCH/tmp"
export APPTAINER_CACHEDIR="$SCRATCH/cache"
export XDG_RUNTIME_DIR="$SCRATCH/runtime"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$XDG_RUNTIME_DIR"
trap 'rm -rf "$SCRATCH"' EXIT

"$PROJECT_ROOT/scripts/build-apptainer.sh"
