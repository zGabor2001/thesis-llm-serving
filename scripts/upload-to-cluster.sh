#!/usr/bin/env bash
set -euo pipefail

# Upload only what the cluster needs: the gateway package, scripts, models,
# requirements.txt and configs. Everything else (.venv, .git, .idea, tests,
# PDFs, Makefile, .env, …) stays local.
#
# Usage:
#   scripts/upload-to-cluster.sh            # upload
#   scripts/upload-to-cluster.sh --dry-run  # show what would be uploaded
#
# Override host/destination with:
#   CLUSTER_HOST=datalab CLUSTER_DEST=thesis-llm-serving scripts/upload-to-cluster.sh

HOST="${CLUSTER_HOST:-datalab}"
DEST="${CLUSTER_DEST:-thesis-llm-serving}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RSYNC_OPTS=(-a -v -P)
DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  RSYNC_OPTS+=(-n)
  DRY_RUN=1
fi

echo "==> Uploading ${LOCAL_ROOT} -> ${HOST}:~/${DEST}/"

rsync "${RSYNC_OPTS[@]}" \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --include='gateway/***' \
  --include='scripts/***' \
  --include='models/***' \
  --include='requirements.txt' \
  --include='configs/***' \
  --exclude='*' \
  "$LOCAL_ROOT/" "${HOST}:~/${DEST}/"

if [ "$DRY_RUN" = "0" ]; then
  echo "==> Ensuring runtime directories exist"
  ssh "$HOST" "mkdir -p ~/${DEST}/run ~/${DEST}/logs ~/${DEST}/traces ~/${DEST}/configs ~/${DEST}/images"

  echo ""
  echo "==> Done. Next steps:"
  echo "    ssh $HOST"
  echo "    cd ~/${DEST}"
  echo "    sbatch scripts/slurm/build-images.sbatch"
fi
