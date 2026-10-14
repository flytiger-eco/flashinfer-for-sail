#!/bin/bash
set -eo pipefail
set -x

cd flashinfer-jit-cache/
export FLASHINFER_CUDA_ARCH_LIST="8.0 8.9"
pip install -U 'setuptools>=77,<82'
python -m build --no-isolation --wheel

WHEEL_FILE=$(ls -t dist/*.whl | head -n 1)
pip install "${WHEEL_FILE}"
python -c "import flashinfer; print('flashinfer', flashinfer.__version__)"
