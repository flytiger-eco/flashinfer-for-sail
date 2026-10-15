#!/bin/bash
set -eo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}/flashinfer-jit-cache"
export FLASHINFER_CUDA_ARCH_LIST="8.0 8.9"
pip install -U 'setuptools>=77,<82'
python -m build --no-isolation --wheel

WHEEL_FILE=$(ls -t dist/*.whl | head -n 1)
pip install "${WHEEL_FILE}"

cd "${ROOT_DIR}/flashinfer-cubin"
python -m build --no-isolation --wheel
python -m pip install dist/*.whl

cd "${ROOT_DIR}"
python -m pip install -v .

python -c "import flashinfer; print('flashinfer', flashinfer.__version__)"