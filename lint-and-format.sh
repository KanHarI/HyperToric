#!/usr/bin/env bash
set -euo pipefail

echo "=== Ruff format ==="
uv run ruff format src/ tests/

echo "=== Ruff check (with autofix) ==="
uv run ruff check --fix src/ tests/

echo "=== mypy ==="
uv run mypy src/

echo "=== All passed ==="
