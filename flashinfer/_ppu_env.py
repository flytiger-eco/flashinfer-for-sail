"""Environment detection for cuda-free (PPU/HGGC) builds.

Deliberately dependency-free: flashinfer.compilation_context evaluates
ppu_sdk_root() at import time (its COMMON_NVCC_FLAGS class body), so this
module must not import anything from flashinfer or flashinfer.jit -- that
re-enters flashinfer.jit.env, which imports compilation_context, closing a
circular import (the failure mode this module was split out to fix).

The canonical semantics live here; flashinfer.jit.sailify_staging re-exports
the accessor for its own callers.
"""

import os
import shutil
from typing import Optional


_DEFAULT_PPU_SDK_ROOT = "/usr/local/PPU_SDK"


def _ppu_sdk_root() -> Optional[str]:
    """Locate the cuda-free PPU SDK install, or None for a plain CUDA build."""
    if os.environ.get("FLASHINFER_JIT_SAILIFY", "1") == "0":
        return None
    # PPU envsetup.sh exports PPU_SDK and PPU_PATH. PPU_SDK_ROOT remains
    # supported as FlashInfer's explicit override.
    for name in ("PPU_SDK_ROOT", "PPU_PATH", "PPU_SDK"):
        root = os.environ.get(name)
        if root and os.path.isdir(root):
            return root
    # The CUDA compatibility SDK may also identify its PPU parent.
    # envsetup.sh ppu exports CUDA_SDK=/usr/local/PPU_SDK/CUDA_SDK
    cuda_sdk = os.environ.get("CUDA_SDK", "")
    if cuda_sdk and "PPU_SDK" in cuda_sdk:
        guess = os.path.dirname(cuda_sdk.rstrip("/"))
        if os.path.isdir(guess):
            return guess

    # Some hermetic CI jobs install the PPU SDK at its canonical location but
    # do not source envsetup.sh. Auto-detect that layout only when no CUDA
    # toolkit was explicitly configured and nvcc is absent (or is the PPU
    # compatibility persona), so real CUDA installations keep their path.
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = shutil.which("nvcc")

    configured_compiler = os.environ.get("FLASHINFER_NVCC")
    configured_hgcc = (
        shutil.which(configured_compiler) or configured_compiler
        if configured_compiler
        else None
    )
    path_hgcc = shutil.which("hgcc")
    for hgcc in (configured_hgcc, path_hgcc):
        if (
            hgcc
            and os.path.basename(hgcc) == "hgcc"
            and os.path.isfile(hgcc)
            and os.access(hgcc, os.X_OK)
        ):
            root = os.path.dirname(os.path.dirname(os.path.realpath(hgcc)))
            # Some PPU installations put an hgcc compatibility persona under
            # CUDA_SDK/bin while the native compiler lives at ../bin/hgcc.
            # Keep $cuda_home at the outer PPU SDK so includes and libhggcrt
            # resolve from the native toolchain layout.
            if os.path.basename(root) == "CUDA_SDK":
                ppu_root = os.path.dirname(root)
                native_hgcc = os.path.join(ppu_root, "bin", "hgcc")
                if os.path.isfile(native_hgcc) and os.access(native_hgcc, os.X_OK):
                    root = ppu_root
            nvcc_is_ppu_persona = nvcc is not None and os.path.realpath(
                nvcc
            ).startswith(root + os.sep)
            if cuda_home is None and (nvcc is None or nvcc_is_ppu_persona):
                return root

    default_hgcc = os.path.join(_DEFAULT_PPU_SDK_ROOT, "bin", "hgcc")
    nvcc_is_ppu_persona = nvcc is not None and os.path.realpath(nvcc).startswith(
        os.path.realpath(_DEFAULT_PPU_SDK_ROOT) + os.sep
    )
    if (
        cuda_home is None
        and (nvcc is None or nvcc_is_ppu_persona)
        and os.path.isfile(default_hgcc)
        and os.access(default_hgcc, os.X_OK)
    ):
        return _DEFAULT_PPU_SDK_ROOT
    return None


def ppu_sdk_root() -> Optional[str]:
    """Public accessor: the cuda-free PPU SDK root, or None for a CUDA build."""
    return _ppu_sdk_root()
