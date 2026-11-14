#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
"${PYTHON:-python3.12}" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
printf '%s\n' 'Ready: .venv/bin/python run_demo.py --prepare-only --output .runs/prepare'
