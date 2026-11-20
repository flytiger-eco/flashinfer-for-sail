#!/bin/bash
# Build the wheel used by the PPU Linux Test (810E) workflow.
#
# Adapted for v0.7.1: upstream restructured flashinfer-jit-cache into a
# provider *shim* package (its wheel build requires
# FLASHINFER_JIT_CACHE_PROVIDER_ARCHS and contains no FlashInfer code),
# so the v0.6.17_dev flow of building and installing the jit-cache wheel
# no longer applies. What this branch actually ships for PPU is the
# root pure-JIT wheel — the same artifact produced by
# scripts/task_ppu_build_wheel.sh for ppu-linux-build.yml — so build
# that and let the workflow stage it from dist/.
set -euo pipefail

# Match the v0.6.18 PPU test environment: the image ships 4.6.0, while
# FlashInfer requires >=4.6.2a0. Keep the PPU stack on the compatible 4.6.x line
# rather than allowing the wheel install to pull newer DSL/MLIR APIs.
python -m pip install 'nvidia-cutlass-dsl==4.6.2'

exec bash "$(dirname "$0")/task_ppu_build_wheel.sh"
