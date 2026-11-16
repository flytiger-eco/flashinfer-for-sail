# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/cpp_extension.py

import functools
import logging
import os
import re
import subprocess
import sys
import sysconfig
from packaging.version import Version
from pathlib import Path
from typing import List, Mapping, Optional

import tvm_ffi
import torch

from . import env as jit_env
from . import sailify_staging
from ..compilation_context import CompilationContext

logger = logging.getLogger(__name__)


def parse_env_flags(env_var_name) -> List[str]:
    env_flags = os.environ.get(env_var_name)
    if env_flags:
        try:
            import shlex

            return shlex.split(env_flags)
        except ValueError as e:
            logger.warning(
                "Could not parse %s with shlex: %s. Falling back to simple split.",
                env_var_name,
                e,
            )
            return env_flags.split()
    return []


def _get_glibcxx_abi_build_flags() -> List[str]:
    glibcxx_abi_cflags = [
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    ]
    return glibcxx_abi_cflags


@functools.cache
def get_cuda_path() -> str:
    # cuda-free (PPU): the only toolkit is the PPU SDK. This must win over
    # CUDA_HOME/`which nvcc` so ninja's $cuda_home always points at the
    # native PPU toolchain and headers.
    ppu_root = sailify_staging.ppu_sdk_root()
    if ppu_root:
        return ppu_root
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is not None:
        return cuda_home
    # get output of "which nvcc"
    nvcc_path = subprocess.run(["which", "nvcc"], capture_output=True)
    if nvcc_path.returncode == 0:
        cuda_home = os.path.dirname(
            os.path.dirname(nvcc_path.stdout.decode("utf-8").strip())
        )
    else:
        cuda_home = "/usr/local/cuda"  # This default value is from: https://github.com/pytorch/pytorch/blob/ceb11a584d6b3fdc600358577d9bf2644f88def9/torch/utils/cpp_extension.py#L115
        if not os.path.exists(cuda_home):
            raise RuntimeError(
                f"Could not find CUDA nvcc or PPU hgcc and default "
                f"{cuda_home=} doesn't exist. Source the PPU SDK envsetup.sh "
                "or set PPU_SDK_ROOT/PPU_PATH for cuda-free builds."
            )
    return cuda_home


@functools.cache
def get_cuda_version() -> Version:
    # Try to query nvcc for CUDA version; if nvcc is unavailable, fall back to torch.version.cuda
    try:
        cuda_home = get_cuda_path()
        nvcc = os.path.join(cuda_home, "bin/nvcc")
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        matches = re.findall(r"release (\d+\.\d+),", txt)
        if not matches:
            raise RuntimeError(
                f"Could not parse CUDA version from nvcc --version output: {txt}"
            )
        return Version(matches[0])
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        # NOTE(Zihao): when nvcc is unavailable, fall back to torch.version.cuda
        if torch.version.cuda is None:
            raise RuntimeError(
                "nvcc not found and PyTorch is not built with CUDA support. "
                "Could not determine CUDA version."
            ) from e
        return Version(torch.version.cuda)


def is_cuda_version_at_least(version_str: str) -> bool:
    return get_cuda_version() >= Version(version_str)


def get_nvcc_parallelism_flags() -> List[str]:
    """Build nvcc flags controlled by FlashInfer parallelism environment variables."""
    env_var_name = "FLASHINFER_NVCC_THREADS"
    default = 1
    value = os.environ.get(env_var_name, str(default))

    try:
        threads = int(value)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using %s.", env_var_name, value, default
        )
        threads = default

    if threads < 1:
        logger.warning("Ignoring %s=%r; value must be >= 1.", env_var_name, value)
        threads = default

    return [f"--threads={threads}"]


@functools.lru_cache(maxsize=1)
def host_compiler_is_gcc() -> bool:
    """Whether the compiler nvcc forwards -Xcompiler flags to is GCC.

    Cached because the underlying torch helper shells out to the compiler.
    """
    from torch.utils.cpp_extension import check_compiler_is_gcc, get_cxx_compiler

    host_compiler = os.environ.get("CC", get_cxx_compiler())
    return check_compiler_is_gcc(host_compiler)


def join_multiline(vs: List[str]) -> str:
    return " $\n    ".join(vs)


def get_cccl_includes() -> List:
    """Get vendored CCCL include directories (added with -I for CTK override precedence)."""
    if sailify_staging.ppu_sdk_root():
        # PPU_SDK provides native cub/libcudacxx/thrust under its include
        # tree. Dropping the vendored NVIDIA CCCL override lets sailify's
        # cuda/std -> hggc/std conversion resolve against those headers.
        return []
    return [p.resolve() for p in jit_env.CCCL_INCLUDE_DIRS]


def get_system_includes(cuda_home: str) -> List:
    """Get list of system include directories."""
    tvm_ffi_include = tvm_ffi.libinfo.find_include_path()
    tvm_ffi_dlpack_include = tvm_ffi.libinfo.find_dlpack_include_path()
    system_includes = [
        sysconfig.get_path("include"),
        "$cuda_home/include",
    ]
    if sailify_staging.ppu_sdk_root():
        # Stock tvm-ffi wheels carry the CUDA vocabulary in
        # tvm/ffi/extra/cuda/* (cuda_runtime.h include, cuda* calls), which
        # has no resolution under a pure PPU toolchain. Stage a sailified
        # copy of the installed header trees ahead of the originals so the
        # converted headers shadow them — stock tvm-ffi and partially
        # converted tvm-ffi-sail installs then use the same build path.
        system_includes += sailify_staging.stage_tvm_ffi_includes(
            tvm_ffi_include, tvm_ffi_dlpack_include
        )
    system_includes += [tvm_ffi_include, tvm_ffi_dlpack_include]
    system_includes += [
        jit_env.FLASHINFER_INCLUDE_DIR.resolve(),
        jit_env.FLASHINFER_CSRC_DIR.resolve(),
    ]
    system_includes += [p.resolve() for p in jit_env.CUTLASS_INCLUDE_DIRS]
    system_includes.append(jit_env.SPDLOG_INCLUDE_DIR.resolve())

    if cuda_home == "/usr":
        # NOTE: this will resolve to /usr/include, which will mess up includes. See #1793
        system_includes.remove("$cuda_home/include")

    return system_includes


def build_common_cflags(
    cuda_home: str,
    extra_include_dirs: Optional[List[Path]] = None,
) -> List[str]:
    """Build common compilation flags."""
    cccl_includes = get_cccl_includes()
    system_includes = get_system_includes(cuda_home)

    common_cflags = []
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        common_cflags.append("-DPy_LIMITED_API=0x03090000")
    common_cflags += _get_glibcxx_abi_build_flags()
    if extra_include_dirs is not None:
        for extra_dir in extra_include_dirs:
            common_cflags.append(f"-I{extra_dir.resolve()}")
    # Vendored CCCL headers use -I (not -isystem) so they take precedence
    # over the CTK-bundled copy. CCCL headers use #pragma system_header
    # internally to suppress warnings. See https://github.com/NVIDIA/cccl/issues/527
    for cccl_dir in cccl_includes:
        common_cflags.append(f"-I{cccl_dir}")
    for sys_dir in system_includes:
        common_cflags.append(f"-isystem {sys_dir}")

    return common_cflags


def build_cflags(
    common_cflags: List[str],
    extra_cflags: Optional[List[str]] = None,
) -> List[str]:
    """Build C++ compilation flags."""
    cflags = [
        "$common_cflags",
        "-fPIC",
    ]
    if extra_cflags is not None:
        cflags += extra_cflags

    # cuda-free (PPU): pull the sailify compat wrapper into every host TU
    if sailify_staging.ppu_sdk_root():
        cflags += ["-include compatible_wrapper.h"]

    env_extra_cflags = parse_env_flags("FLASHINFER_EXTRA_CFLAGS")
    if env_extra_cflags is not None:
        cflags += env_extra_cflags

    return cflags


def build_cuda_cflags(
    common_cflags: List[str],
    extra_cuda_cflags: Optional[List[str]] = None,
) -> List[str]:
    """Build CUDA compilation flags."""
    cuda_cflags: List[str] = []
    cc_env = os.environ.get("CC")
    if cc_env is not None:
        cuda_cflags += ["-ccbin", cc_env]
    cuda_cflags += [
        "$common_cflags",
        "--compiler-options=-fPIC",
        "--expt-relaxed-constexpr",
    ]
    cuda_version = get_cuda_version()
    # enable -static-global-template-stub when cuda version >= 12.8
    if cuda_version >= Version("12.8"):
        cuda_cflags += [
            "-static-global-template-stub=false",
        ]

    cpp_ext_initial_compilation_context = CompilationContext()
    global_flags = cpp_ext_initial_compilation_context.get_nvcc_flags_list(
        map_sm107_to_100f=True
    )
    if extra_cuda_cflags is not None:
        # Check if module provides architecture flags
        module_has_gencode = any(
            flag.startswith("-gencode=") for flag in extra_cuda_cflags
        )

        if module_has_gencode:
            # Use module's architecture flags, but keep global non-architecture flags
            global_non_arch_flags = [
                flag for flag in global_flags if not flag.startswith("-gencode=")
            ]
            cuda_cflags += global_non_arch_flags + extra_cuda_cflags
        else:
            # No module architecture flags, use both global and module flags
            cuda_cflags += global_flags + extra_cuda_cflags
    else:
        # No module flags, use global flags
        cuda_cflags += global_flags

    # cuda-free (PPU): the native hggc headers (hggc_runtime_api.h etc.)
    # and the compat wrapper; the staged .ppu_compat dir already precedes
    # these via extra_include_dirs, so its shims keep precedence.
    ppu_root = sailify_staging.ppu_sdk_root()
    if ppu_root:
        cuda_cflags += [
            f"-I{ppu_root}/include",
            "-include compatible_wrapper.h",
            # tvm-ffi (PPU build, tvm-ffi-sail) gates its hggc runtime face
            # (<hggc_runtime.h>) behind this macro instead of
            # <cuda_runtime.h>; the compat bridge would cover the latter,
            # but the native face keeps tvm-ffi on the PPU path.
            "-DTVM_FFI_CUBIN_LAUNCHER_USE_PPU",
        ]

    env_extra_cuda_cflags = parse_env_flags("FLASHINFER_EXTRA_CUDAFLAGS")
    if env_extra_cuda_cflags is not None:
        cuda_cflags += env_extra_cuda_cflags

    # hgcc consumes the PPU-native option vocabulary. Keep module flag
    # producers CUDA-compatible and translate only at the toolchain boundary.
    if ppu_root:
        cuda_cflags = _convert_nvcc_flags_to_hgcc(cuda_cflags)

    return cuda_cflags


@functools.cache
def _hgcc_flag_converter():
    """Build a converter from sailify's canonical nvcc-to-hgcc option table."""
    try:
        NVCC_HGCC_OPTIONS = sailify_staging.import_sailify(
            "nvcc_hgcc_options"
        ).NVCC_HGCC_OPTIONS
    except ImportError:
        logger.warning(
            "sailify.nvcc_hgcc_options unavailable; passing nvcc flags to hgcc"
        )
        return None

    flag_map: dict = {}
    for long_name, entry in NVCC_HGCC_OPTIONS.items():
        flag_map[long_name] = entry
        short = entry.get("short")
        if short:
            flag_map[short] = entry

    def cuda_arch_to_ppu(arch: str) -> str:
        if arch.startswith("ppu_"):
            return arch
        match = re.match(r"(\d+)\.(\d+)", arch.split("+")[0].rstrip("a"))
        if match is None:
            raise ValueError(f"Invalid CUDA architecture: {arch!r}")
        major, minor = int(match.group(1)), int(match.group(2))
        if major == 8 and minor in (0, 6):
            return "ppu_10"
        if (major == 8 and minor == 9) or major == 9:
            return "ppu_15"
        return "ppu_10"

    def convert(flags: List[str]) -> List[str]:
        result: List[str] = []
        ppu_arches: List[str] = []
        i = 0
        while i < len(flags):
            flag = flags[i]
            opt, eq, val = flag.partition("=")
            if opt in ("-gencode", "--gpu-code", "--gpu-architecture", "-arch"):
                if eq:
                    arch_val = val
                elif i + 1 < len(flags):
                    arch_val = flags[i + 1]
                    i += 1
                else:
                    i += 1
                    continue
                if opt in ("-gencode", "--gpu-code"):
                    match = re.search(r"(?:sm_|compute_)(\d+)(\d)", arch_val)
                    if match is None:
                        raise ValueError(f"Invalid gencode value: {arch_val!r}")
                    arch_val = f"{match.group(1)}.{match.group(2)}"
                else:
                    match = re.match(r"(?:sm_|compute_)(\d+)(\d)", arch_val)
                    if match is not None:
                        arch_val = f"{match.group(1)}.{match.group(2)}"
                ppu_arches.append(cuda_arch_to_ppu(arch_val))
                i += 1
                continue

            entry = flag_map.get(opt)
            if entry is None:
                result.append(flag)
                i += 1
                continue
            if entry.get("needs_ignore"):
                if entry.get("needs_value") and not eq and i + 1 < len(flags):
                    i += 2
                else:
                    i += 1
                continue
            mapped = entry.get("map_to")
            if mapped is None:
                result.append(flag)
                i += 1
                continue
            mapped_list = mapped if isinstance(mapped, list) else [mapped]
            if not entry.get("needs_value"):
                result.extend(mapped_list)
                i += 1
                continue
            if eq:
                raw_value = val
            elif i + 1 < len(flags):
                raw_value = flags[i + 1]
                i += 1
            else:
                raw_value = ""
            value_map = entry.get("value_map")
            if value_map and raw_value in value_map:
                mapped_value = value_map[raw_value]
                if isinstance(mapped_value, dict) and mapped_value.get("needs_ignore"):
                    i += 1
                    continue
                result.extend(mapped_list)
                result.append(str(mapped_value))
            else:
                result.extend(mapped_list)
                if raw_value:
                    result.append(raw_value)
            i += 1

        if ppu_arches:
            result.append(
                f"--gpu-architecture={','.join(dict.fromkeys(ppu_arches))}"
            )
        return result

    return convert


def _convert_nvcc_flags_to_hgcc(flags: List[str]) -> List[str]:
    """Convert nvcc-style flags to hgcc flags when sailify is available."""
    convert = _hgcc_flag_converter()
    return flags if convert is None else convert(flags)


def generate_ninja_build_for_op(
    name: str,
    sources: List[Path],
    extra_cflags: Optional[List[str]],
    extra_cuda_cflags: Optional[List[str]],
    extra_ldflags: Optional[List[str]],
    extra_include_dirs: Optional[List[Path]],
    needs_device_linking: bool = False,
    embedded_cubins: Optional[Mapping[str, Path]] = None,
    extra_cuda_cflags_by_source: Optional[Mapping[Path, List[str]]] = None,
) -> str:
    cuda_home = get_cuda_path()
    common_cflags = build_common_cflags(cuda_home, extra_include_dirs)
    cflags = build_cflags(common_cflags, extra_cflags)
    cuda_cflags = build_cuda_cflags(common_cflags, extra_cuda_cflags)
    cuda_arch_flags = [flag for flag in cuda_cflags if flag.startswith("-gencode=")]
    cuda_cflags_by_source = {
        Path(source).resolve(): build_cuda_cflags(common_cflags, flags)
        for source, flags in (extra_cuda_cflags_by_source or {}).items()
    }

    ldflags = [
        "-shared",
        "-L$cuda_home/lib64",
        "-L$cuda_home/lib64/stubs",
        "-lcudart",
        "-lcuda",
    ]

    # cuda-free (PPU): hgcc emits only the hggc runtime ABI. Do not retain
    # CUDA linker paths or libraries, which could resolve from ldconfig and
    # silently reintroduce a CUDA-persona dependency.
    ppu_root = sailify_staging.ppu_sdk_root()
    if ppu_root:
        ldflags = [
            "-shared",
            f"-L{ppu_root}/lib",
            "-lhggcrt1",
        ]

    env_extra_ldflags = parse_env_flags("FLASHINFER_EXTRA_LDFLAGS")
    if env_extra_ldflags is not None:
        ldflags += env_extra_ldflags

    if extra_ldflags is not None:
        ldflags += extra_ldflags

    cxx = os.environ.get("CXX", "c++")
    # The nvcc persona emits __cuda* registration symbols; native hgcc emits
    # the __hggc* ABI provided by libhggcrt.
    nvcc_default = "$cuda_home/bin/hgcc" if ppu_root else "$cuda_home/bin/nvcc"
    nvcc = os.environ.get("FLASHINFER_NVCC", nvcc_default)
    # Compiler launchers (e.g., sccache, ccache) — empty string when unset
    cxx_launcher = os.environ.get("FLASHINFER_CXX_LAUNCHER", "")
    nvcc_launcher = os.environ.get("FLASHINFER_NVCC_LAUNCHER", "")

    lines = [
        "ninja_required_version = 1.3",
        f"name = {name}",
        f"cuda_home = {cuda_home}",
        f"cxx = {cxx}",
        f"nvcc = {nvcc}",
        f"cxx_launcher = {cxx_launcher}",
        f"nvcc_launcher = {nvcc_launcher}",
        "",
        "common_cflags = " + join_multiline(common_cflags),
        "cflags = " + join_multiline(cflags),
        "post_cflags =",
        "cuda_cflags = " + join_multiline(cuda_cflags),
        "cuda_post_cflags =",
        "cuda_arch_flags = " + join_multiline(cuda_arch_flags),
        "ldflags = " + join_multiline(ldflags),
        "",
        "rule compile",
        "  command = $cxx_launcher $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
        "rule cuda_compile",
        "  command = $nvcc_launcher $nvcc --generate-dependencies-with-compile -MF $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
    ]

    if embedded_cubins:
        lines.extend(
            [
                "rule embed_cubin",
                f"  command = {sys.executable} -m tvm_ffi.utils.embed_cubin "
                "--output-obj $out --input-obj $in --cubin $cubin --name $cubin_name",
                "",
            ]
        )

    # Add nvcc linking rule for device code
    if needs_device_linking:
        lines.extend(
            [
                "rule nvcc_link",
                "  command = $nvcc -shared $cuda_arch_flags $in $ldflags -o $out",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "rule link",
                "  command = $cxx $in $ldflags -o $out",
                "",
            ]
        )

    # Use absolute paths for outputs so ninja files work with any workdir
    # This enables isolated workdirs for runtime JIT (avoiding .ninja_log races)
    # while still supporting subninja for parallel AOT builds
    output_dir = jit_env.FLASHINFER_JIT_DIR / name

    objects = []
    for source in sources:
        is_cuda = source.suffix == ".cu"
        object_suffix = ".cuda.o" if is_cuda else ".o"
        cmd = "cuda_compile" if is_cuda else "compile"
        obj_name = f"{source.parent.name}_{source.stem}{object_suffix}"
        obj = str((output_dir / obj_name).resolve())
        objects.append(obj)
        lines.append(f"build {obj}: {cmd} {source.resolve()}")
        if source.resolve() in cuda_cflags_by_source:
            lines.append(
                "  cuda_cflags = "
                + join_multiline(cuda_cflags_by_source[source.resolve()])
            )

    if embedded_cubins:
        if not objects:
            raise ValueError("embedded cubins require at least one host object")
        current_obj = objects[0]
        for index, (cubin_name, cubin_path) in enumerate(
            sorted(embedded_cubins.items())
        ):
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cubin_name) is None:
                raise ValueError(f"invalid embedded cubin identifier: {cubin_name!r}")
            resolved_cubin = Path(cubin_path).resolve(strict=True)
            embedded_obj = str(
                (output_dir / f"embedded_{index}_{Path(current_obj).name}").resolve()
            )
            lines.extend(
                [
                    f"build {embedded_obj}: embed_cubin {current_obj} | {resolved_cubin}",
                    f"  cubin = {resolved_cubin}",
                    f"  cubin_name = {cubin_name}",
                ]
            )
            current_obj = embedded_obj
        objects[0] = current_obj

    lines.append("")
    link_rule = "nvcc_link" if needs_device_linking else "link"
    output_so = str((output_dir / f"{name}.so").resolve())
    lines.append(f"build {output_so}: {link_rule} " + " ".join(objects))
    lines.append(f"default {output_so}")
    lines.append("")

    return "\n".join(lines)


def _get_num_workers() -> Optional[int]:
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs is not None and max_jobs.isdigit():
        return int(max_jobs)
    return None


def run_ninja(
    workdir: Path,
    ninja_file: Path,
    verbose: bool,
    max_jobs: Optional[int] = None,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "ninja",
        "-v",
        "-C",
        str(workdir.resolve()),
        "-f",
        str(ninja_file.resolve()),
    ]
    num_workers = max_jobs if max_jobs is not None else _get_num_workers()
    if num_workers is not None:
        command += ["-j", str(num_workers)]

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        subprocess.run(
            command,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workdir.resolve()),
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = "Ninja build failed."
        if e.output:
            msg += " Ninja output:\n" + e.output
        raise RuntimeError(msg) from e
