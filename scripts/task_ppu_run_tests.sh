#!/usr/bin/env bash
set -eu

cd "${PPU_SOURCE_DIR:-/workspace/source}"

echo '=== PPU Verification ==='
echo "Hostname: $(hostname), Node: ${NODE_NAME:-}"
echo "RANK=${RANK:-} NODE_RANK=${NODE_RANK:-} WORLD_SIZE=${WORLD_SIZE:-} NPROC_PER_NODE=${NPROC_PER_NODE:-}"
ppu-smi || true
echo '--- source tree ---'
ls -al


WHEEL=$(ls -t wheelhouse/*/*.whl 2>/dev/null | head -n 1 || true)
if [ -z "$WHEEL" ]; then
  echo "ERROR: no wheel found under wheelhouse/" >&2
  find wheelhouse -type f -print || true
  exit 1
fi
echo "Installing wheel: $WHEEL"
pip install --no-deps --force-reinstall "$WHEEL"

python -c "import flashinfer; print('flashinfer', flashinfer.__version__)"
python -m flashinfer show-config || true


python -c "import pytest" 2>/dev/null || pip install --no-deps pytest


DESELECT=(
  --deselect "tests/utils/test_norm.py::test_layernorm_quant"
  --deselect "tests/utils/test_norm.py::test_layernorm_quant_invalid_inputs"
  --deselect "tests/utils/test_sampling.py::test_sampling_from_logits_freq[normal_distribution(std=1)-32000]"
  --deselect "tests/utils/test_sampling.py::test_sampling_from_logits_freq[normal_distribution(std=1)-128256]"
  --deselect "tests/utils/test_sampling.py::test_sampling_from_logits_seed_offset_reproducibility[32000-99]"
  --deselect "tests/utils/test_sampling.py::test_sampling_from_logits_seed_offset_reproducibility[32000-989]"
  --deselect "tests/utils/test_sampling.py::test_sampling_from_logits_seed_offset_reproducibility[128256-99]"
  --deselect "tests/utils/test_sampling.py::test_sampling_from_logits_seed_offset_reproducibility[128256-989]"
)


echo '=== Run unit tests (PPU subset) ==='
set +e
pytest --continue-on-collection-errors \
  -v -ra --tb=short --color=yes \
  "${DESELECT[@]}" \
  tests/test_jit_cpp_ext.py \
  tests/autotuner/test_autotuner_configs.py \
  tests/grouped_mm/test_grouped_mm_bf16.py \
  tests/utils/test_norm.py \
  tests/utils/test_activation.py \
  tests/utils/test_sampling.py
rc=$?
set -e
echo "=== DONE (pytest exit code: ${rc}) ==="
exit "${rc}"
