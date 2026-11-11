#!/usr/bin/env python3
"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Generate minimal loadable CUDA ABI stub libraries for cuda-free builds.

PyTorch-for-SAIL links against the CUDA ABI (libcuda.so.1, libcublas.so.13,
libcublasLt.so.13, libcudnn.so.8, ...). On hosts without the PPU CUDA
compatibility SDK (PPU_SDK/CUDA_SDK) or an NVIDIA driver, ``import torch``
fails at dlopen time even though the cuda-free FlashInfer toolchain (sailify
+ native hgcc) never calls into those libraries: kernel builds link
libhggcrt1 only. This script scans the installed torch libraries for
undefined CUDA ABI symbols and compiles versioned no-op stub .so files that
satisfy the dynamic loader.

The stubs are for build-time bootstrap only — never install them on a host
that runs real CUDA/PPU workloads, and never ship them in a wheel.
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# SONAME -> symbol prefix handled by that stub. Order matters where prefixes
# overlap (cublasLt before cublas): the first matching prefix wins, keeping
# the assignment mutually exclusive.
STUB_LIBS = [
    ("libcublasLt.so.13", re.compile(r"^cublasLt")),
    ("libcublas.so.13", re.compile(r"^cublas")),
    ("libcudnn.so.8", re.compile(r"^cudnn")),
    ("libcuda.so.1", re.compile(r"^cu[A-Z]")),
]

# Only generated with --with-cudart (hosts without the nvidia-cuda-runtime
# wheel): the CUDA runtime API prefix.
CUDART_LIB = ("libcudart.so.13", re.compile(r"^cuda"))


def find_torch_lib_dir() -> Path:
    """Locate site-packages/torch/lib without importing torch.

    The stubs exist to make ``import torch`` possible in the first place, so
    this must not depend on it.
    """
    import sysconfig

    purelib = Path(sysconfig.get_paths()["purelib"])
    for candidate in (purelib / "torch" / "lib",):
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"torch/lib not found under {purelib}")


def collect_undefined_symbols(torch_lib_dir: Path) -> set:
    """Union of undefined dynamic symbols across the installed torch libs."""
    syms = set()
    for lib in sorted(torch_lib_dir.glob("*.so*")):
        result = subprocess.run(
            ["nm", "-D", "--undefined-only", str(lib)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                syms.add(parts[-1])
    if not syms:
        raise RuntimeError(f"no undefined symbols found under {torch_lib_dir}")
    return syms


def group_symbols(syms: set, extra_libs: list) -> dict:
    """Map SONAME -> {version_node | None: [symbols]}.

    ``sym@VER`` references land in version node VER (the stub must define
    that version or the loader aborts with a dl-lookup assertion);
    unversioned references go to the anonymous base node. Each symbol is
    assigned to the first matching prefix so overlapping prefixes
    (cublas vs cublasLt) stay mutually exclusive.
    """
    per_lib = defaultdict(lambda: defaultdict(list))
    patterns = STUB_LIBS + extra_libs
    for raw in sorted(syms):
        sym, _, ver = raw.partition("@")
        for soname, prefix in patterns:
            if prefix.match(sym):
                per_lib[soname][ver if ver else None].append(sym)
                break
    return per_lib


def build_stub(out_dir: Path, soname: str, versions: dict) -> None:
    """Compile one stub .so with a matching version script."""
    all_syms = sorted({s for syms in versions.values() for s in syms})
    c_src = "\n".join(f"long {s}(void) {{ return 0; }}" for s in all_syms) + "\n"
    script_nodes = []
    for ver in sorted(versions, key=lambda v: (v is None, v)):
        syms = versions[ver]
        node = ver if ver is not None else ""
        script_nodes.append(
            f"{node} {{\n  global:\n    " + ";\n    ".join(syms) + ";\n};"
        )
    c_file = out_dir / f"stub_{soname}.c"
    map_file = out_dir / f"{soname}.map"
    c_file.write_text(c_src)
    map_file.write_text("\n".join(script_nodes) + "\n")
    subprocess.run(
        [
            "gcc",
            "-shared",
            "-fPIC",
            "-o",
            str(out_dir / soname),
            f"-Wl,-soname,{soname}",
            f"-Wl,--version-script,{map_file}",
            str(c_file),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".cuda_abi_stubs"),
        help="output directory for the stub libraries (default: .cuda_abi_stubs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the stubs already exist",
    )
    parser.add_argument(
        "--with-cudart",
        action="store_true",
        help="also stub libcudart.so.13 (only for hosts without the "
        "nvidia-cuda-runtime wheel)",
    )
    args = parser.parse_args()

    stub_names = [soname for soname, _ in STUB_LIBS]
    if args.with_cudart:
        stub_names.append(CUDART_LIB[0])
    existing = [(args.out / n).is_file() for n in stub_names]
    if all(existing) and not args.force:
        print(f"cuda ABI stubs already present in {args.out}; use --force to rebuild")
        return

    if subprocess.run(["which", "gcc"], capture_output=True).returncode != 0:
        raise RuntimeError("gcc is required to build the stub libraries")

    torch_lib_dir = find_torch_lib_dir()
    syms = collect_undefined_symbols(torch_lib_dir)
    extra = [CUDART_LIB] if args.with_cudart else []
    per_lib = group_symbols(syms, extra)

    args.out.mkdir(parents=True, exist_ok=True)
    for soname, versions in sorted(per_lib.items()):
        build_stub(args.out, soname, versions)
        n = sum(len(v) for v in versions.values())
        nodes = ", ".join(str(v) for v in versions if v is not None) or "unversioned"
        print(f"built {args.out / soname}: {n} symbols ({nodes})")
    print(
        f"done. prepend to LD_LIBRARY_PATH:\n"
        f"  export LD_LIBRARY_PATH={args.out.resolve()}:$LD_LIBRARY_PATH"
    )


if __name__ == "__main__":
    main()
