#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python main.py -d TAIL -m moram_vit_b16 zero_shot True "$@"
