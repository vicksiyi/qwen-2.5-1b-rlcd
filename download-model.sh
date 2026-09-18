#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
unset HF_HUB_OFFLINE
exec .venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("mlx-community/Qwen2.5-1.5B-Instruct-4bit", revision="8b403126fc14f14cfc99bb4cfa72ecbc129ea677", local_dir="models/qwen2.5-1.5b-instruct-4bit")'
