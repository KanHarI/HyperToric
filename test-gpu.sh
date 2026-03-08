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

VALID_ARCHS=(cpu metal vulkan cuda)

is_valid_arch() {
    for valid in "${VALID_ARCHS[@]}"; do
        [[ "$1" == "$valid" ]] && return 0
    done
    return 1
}

ARCH="${1:-metal}"

if [[ "$ARCH" == "all" ]]; then
    shift
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

if is_valid_arch "$ARCH"; then
    shift
else
    # First arg is not an arch — treat it as a pytest arg, default to metal
    ARCH="metal"
fi

echo "=== Running tests on ${ARCH} backend ==="
uv run pytest -m "not slow" -v --ti-arch "${ARCH}" "$@"
