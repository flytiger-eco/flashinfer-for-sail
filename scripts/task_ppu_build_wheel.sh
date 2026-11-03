#!/bin/bash
# Build the flashinfer-for-sail wheel (pure JIT, no AOT pre-compile) inside the
# PPU build container.
#
# ci/bash.sh mounts the checked-out repo (with submodules) at /workspace and runs
# with -w /workspace, so the current directory is the repository root and the
# resulting dist/*.whl lands back on the host workspace for artifact upload.
#
# Pure JIT: the wheel ships the flashinfer Python package, its JIT runtime and
# the kernel *sources*/headers (csrc, include, cutlass/spdlog/cccl) — there is NO
# ahead-of-time kernel pre-compilation. Kernels are compiled on first use at
# runtime. build_wheel() only stages sources via _create_data_dir, so the old
# `python -m flashinfer.aot` step merely filled the throwaway runtime workspace
# cache and is dropped here (same wheel, far less build time).
#
# BUILD_NVEP=0 / FLASHINFER_BUILD_NO_PIP=1 turn off the NVEP (NCCL/NIXL endpoint)
# build and build-time pip installs, so `python -m build --wheel` performs no
# heavy compilation while producing the wheel.
set -eo pipefail
set -x

: "${TORCH_CUDA_ARCH_LIST:=8.0}"
: "${FLASHINFER_CUDA_ARCH_LIST:=8.0}"
export TORCH_CUDA_ARCH_LIST
export FLASHINFER_CUDA_ARCH_LIST

echo "========================================"
echo "Build environment"
echo "========================================"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "FLASHINFER_CUDA_ARCH_LIST: ${FLASHINFER_CUDA_ARCH_LIST}"
python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
nvcc --version || echo "nvcc not found"
echo "  - Available Memory: $(free -g | awk '/^Mem:/ {print $7}') GB"
echo "  - Number of Processors: $(nproc)"

echo ""
echo "========================================"
echo "Installing build dependencies"
echo "========================================"
pip install packaging setuptools build pyproject_hooks --force-reinstall --no-deps
pip install --upgrade filelock
pip install tomli
pip install apache-tvm-ffi
pip install pynvml==13.0.1

echo ""
echo "========================================"
echo "Building wheel (pure JIT, no AOT pre-compile)"
echo "========================================"
rm -rf dist
SECONDS=0
rc=0
FLASHINFER_BUILD_NO_PIP=1 BUILD_NVEP=0 python -m build --no-isolation --wheel 2>&1 | tee ppu_wheel_build.log || rc=$?
echo "[build] wheel 构建耗时 ${SECONDS}s，退出码 ${rc}"
[ "${rc}" -eq 0 ] || exit "${rc}"

WHEEL_FILE=$(ls -t dist/*.whl | head -n 1)
echo "Built wheel: ${WHEEL_FILE}"

echo ""
echo "========================================"
echo "Installing and verifying wheel"
echo "========================================"
pip install "${WHEEL_FILE}"
python -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

echo ""
echo "========================================"
echo "Build succeeded: ${WHEEL_FILE}"
echo "========================================"
ls -lh dist/
