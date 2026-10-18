#!/usr/bin/env bash
set -euo pipefail

# Download the base model used by the serving stack.
#
# The base model is a GPTQ 4-bit Qwen2.5-0.5B-Instruct (see models/base/config.json).
# The tenant LoRA adapters (models/loras/tenant-*) are local artifacts trained against
# "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit" (see models/loras/*/adapter_config.json);
# they are NOT on the HuggingFace Hub and must be copied or re-trained separately.
#
# Requires the HuggingFace CLI:  pip install 'huggingface_hub[cli]'
# (installed by `make setup`).
#
# Usage:
#   scripts/fetch-models.sh            # skip if already present
#   scripts/fetch-models.sh --force    # re-download
#   BASE_REPO=<hf-repo> scripts/fetch-models.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-$PROJECT_ROOT/models}"
BASE_REPO="${BASE_REPO:-Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4}"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# Load HF_TOKEN from .env if present.
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_ROOT/.env"
  set +a
fi

if command -v hf >/dev/null 2>&1; then
  HF_BIN="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_BIN="huggingface-cli"
else
  echo "ERROR: 'hf'/'huggingface-cli' not found." >&2
  echo "       Install with:  pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi

DEST="$MODEL_DIR/base"

if [ "$FORCE" = "0" ] && [ -f "$DEST/config.json" ]; then
  echo "Base model already present at $DEST (pass --force to re-download)."
else
  echo "==> Downloading base model: $BASE_REPO"
  mkdir -p "$DEST"
  if [ -n "${HF_TOKEN:-}" ]; then
    "$HF_BIN" download "$BASE_REPO" --local-dir "$DEST" --token "$HF_TOKEN"
  else
    "$HF_BIN" download "$BASE_REPO" --local-dir "$DEST"
  fi
fi

echo ""
echo "LoRA adapters (tenant-a/b/c) are fixed input artifacts — see"
echo "$MODEL_DIR/loras/README.md for provenance."
