#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="$PWD/.cache/huggingface"
export MODEL_ID="$PWD/models/qwen2.5-1.5b-instruct-4bit"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port "${PORT:-8000}"
