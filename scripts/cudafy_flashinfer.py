#!/usr/bin/env python3
"""Prepare a t-head FlashInfer v0.6.18-derived checkout for CUDA wheel builds."""

import argparse
import re
import subprocess
from pathlib import Path

# NVIDIA CUTLASS gitlink in flashinfer-ai/flashinfer v0.6.18.
CUTLASS_COMMIT = "b46b16d003484063bca4ed365e44095c4c6ed633"
CUTLASS_REPO = "https://github.com/NVIDIA/cutlass.git"


def cudafy(source: Path) -> None:
    version = (source / "version.txt").read_text().strip()
    if version != "0.6.18":
        raise SystemExit(f"Unsupported FlashInfer version {version!r}; expected 0.6.18")

    rewrites = {
        "pyproject.toml": [
            ('"3rdparty/actlize"', '"3rdparty/cutlass"'),
            ('"flashinfer.data.sailify" = "3rdparty/sailify"\n', ""),
        ],
        "build_backend.py": [
            ('ln("3rdparty/actlize", "cutlass")', 'ln("3rdparty/cutlass", "cutlass")'),
            ('    ln("3rdparty/sailify", "sailify")\n', ""),
        ],
        "flashinfer/aot.py": [
            ('"3rdparty" / "actlize"', '"3rdparty" / "cutlass"'),
        ],
        "flashinfer/jit/core.py": [
            (
                '    USE_SAIL = "PPU_SDK" in os.environ.keys()\n'
                "    if USE_SAIL:\n"
                '        cflags.append("-DUSE_SAIL")\n'
                '        cuda_cflags.append("-DUSE_SAIL")\n',
                "",
            ),
        ],
        "flashinfer/jit/attention/modules.py": [
            (
                '        # For PPU Compatibility\n        and "PPU_SDK" not in os.environ.keys()\n',
                "",
            ),
        ],
    }
    converted = {}
    for filename, replacements in rewrites.items():
        path = source / filename
        content = path.read_text()
        for old, new in replacements:
            expected = 2 if filename == "flashinfer/aot.py" else 1
            if content.count(old) != expected:
                raise SystemExit(f"{path}: unsupported source layout for {old.strip()}")
            content = content.replace(old, new)
        if filename == "pyproject.toml":
            content, count = re.subn(
                r"^# sailify \(CUDA-to-PPU source converter\)[^\n]*\n"
                r"^# [^\n]*\n"
                r'^"flashinfer\.data\.sailify" = \[\n.*?^\]\n',
                "",
                content,
                flags=re.MULTILINE | re.DOTALL,
            )
            if count != 1:
                raise SystemExit(f"{path}: missing Sailify package data")
        converted[path] = content

    # Keep CUDA mode enabled even when the compatibility image exports PPU SDK
    # variables. The existing JIT code selects NVCC, CUDA libraries and raw
    # sources when this detector returns None.
    ppu_env = source / "flashinfer/_ppu_env.py"
    if "def ppu_sdk_root() -> Optional[str]:" not in ppu_env.read_text():
        raise SystemExit(f"{ppu_env}: unsupported PPU toolchain detector")
    converted[ppu_env] = (
        '"""Toolchain selection for the CUDA-only wheel."""\n\n\n'
        "def ppu_sdk_root() -> None:\n"
        "    return None\n"
    )
    for path, content in converted.items():
        if path.suffix == ".py":
            compile(content, str(path), "exec")

    cutlass = source / "3rdparty/cutlass"
    if cutlass.exists() or cutlass.is_symlink():
        raise SystemExit(f"{cutlass}: expected a fresh t-head checkout")
    cutlass.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=cutlass, check=True)
    subprocess.run(
        ["git", "fetch", "--depth", "1", CUTLASS_REPO, CUTLASS_COMMIT],
        cwd=cutlass,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "--detach", CUTLASS_COMMIT],
        cwd=cutlass,
        check=True,
    )
    for path, content in converted.items():
        path.write_text(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="fresh t-head source checkout")
    cudafy(parser.parse_args().source.resolve())
