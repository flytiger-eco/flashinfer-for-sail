"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .sailify_staging import sailify_enabled


def gen_topk_module() -> JitSpec:
    sources = [
        jit_env.FLASHINFER_CSRC_DIR / "topk.cu",
        # cub_topk.cu needs the CCCL 2.9 execution-environment headers
        # (<cuda/__execution/*.h>) that the cuda-free PPU toolchain does not
        # ship; stage link-satisfying stubs instead so the module (and the
        # default radix backend) still builds on PPU.
        jit_env.FLASHINFER_CSRC_DIR
        / (
            "cub_topk_ppu_stub.cu"
            if sailify_enabled()
            else "cub_topk.cu"
        ),
        jit_env.FLASHINFER_CSRC_DIR / "flashinfer_topk_binding.cu",
        jit_env.FLASHINFER_CSRC_DIR / "flashinfer_fast_topk_clusters_binding.cu",
    ]
    return gen_jit_spec(
        "topk",
        sources,
        extra_cuda_cflags=["-lineinfo"],
    )
