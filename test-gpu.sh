#!/usr/bin/env bash
# Run tests against a specific Taichi backend.
# Usage: ./test-gpu.sh [arch] [pytest args...]
#   ./test-gpu.sh                  # run on Metal (default)
#   ./test-gpu.sh vulkan           # run on Vulkan
#   ./test-gpu.sh cuda             # run on CUDA (cloud GPUs)
#   ./test-gpu.sh cpu              # run on CPU
#   ./test-gpu.sh metal -k test_fields  # field tests only on Metal
#   ./test-gpu.sh all              # run on all available backends sequentially
set -euo pipefail

ARCH="${1:-metal}"
VALID_ARCHS=(cpu metal vulkan cuda)

if [[ "$ARCH" == "all" ]]; then
    shift || true
    for a in "${VALID_ARCHS[@]}"; do
        echo ""
        echo "=== Testing on ${a} backend ==="
        uv run pytest -m "not slow" -v --ti-arch "${a}" "$@" || {
            echo "=== FAILED on ${a} ==="
            exit 1
        }
    done
    echo ""
    echo "=== All backends passed ==="
    exit 0
fi

for valid in "${VALID_ARCHS[@]}"; do
    if [[ "$ARCH" == "$valid" ]]; then
        shift || true
        break
    fi
done
# If no valid arch matched, default to metal and don't shift
if [[ "$ARCH" != "cpu" && "$ARCH" != "metal" && "$ARCH" != "vulkan" && "$ARCH" != "cuda" ]]; then
    ARCH="metal"
fi

echo "=== Running tests on ${ARCH} backend ==="
uv run pytest -m "not slow" -v --ti-arch "${ARCH}" "$@"
