<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-black-background.png?raw=true">
    <img alt="FlashInfer" src="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-white-background.png?raw=true" width=55%>
  </picture>
</p>
<h1 align="center">
</h1>

<p align="center">
| <a href="https://docs.flashinfer.ai"><b>Documentation</b></a> | <a href="https://github.com/flashinfer-ai/flashinfer/releases/latest"><b>Latest Release</b></a> | <a href="https://flashinfer.ai"><b>Blog</b></a> | <a href="https://join.slack.com/t/flashinfer/shared_invite/zt-379wct3hc-D5jR~1ZKQcU00WHsXhgvtA"><b>Slack</b></a> |  <a href="https://github.com/orgs/flashinfer-ai/discussions"><b>Discussion Forum</b></a> |
</p>

[![Build Status](https://ci.tlcpack.ai/job/flashinfer-ci/job/main/badge/icon)](https://ci.tlcpack.ai/job/flashinfer-ci/job/main/)
[![Documentation](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml/badge.svg)](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml)

**FlashInfer-for-SAIL** is a PPU-adapted kernel library and generator based on FlashInfer v0.6.12. It provides runtime dependency, backend kernel, and build workflow adaptations for T-Head AI accelerator chips. It keeps FlashInfer's high-performance kernel generation capabilities and integrates PPU platform optimizations for running large language models and multimodal models on PPU devices.

This document only covers the basic installation, verification, and usage workflow. For Attention Backend, JIT/AOT build tuning, known issues, and model-specific instructions, see the FlashInfer-for-SAIL User Guide.

## Why FlashInfer?

- **State-of-the-art Performance**: Optimized kernels for prefill, decode, and mixed batching scenarios
- **Multiple Backends**: Automatically selects the best backend for your hardware and workload
- **Modern Architecture Support**: Support for ppu001/ppu0015
- **Production-Ready**: CUDAGraph and torch.compile compatible for low-latency serving

## Core Features

### Attention Kernels
- **Paged and Ragged KV-Cache**: Efficient memory management for dynamic batch serving
- **Decode, Prefill, and Append**: Optimized kernels for all attention phases
- **MLA Attention**: Native support for DeepSeek's Multi-Latent Attention
- **Cascade Attention**: Memory-efficient hierarchical KV-Cache for shared prefixes
- **Sparse Attention**: Block-sparse and variable block-sparse patterns
- **POD-Attention**: Fused prefill+decode for mixed batching

### Sampling & Decoding
- **Sorting-Free Sampling**: Efficient Top-K, Top-P, and Min-P without sorting
- **Speculative Decoding**: Chain speculative sampling support

### Other Operators
- **RoPE**: LLaMA-style rotary position embeddings (including LLaMA 3.1)
- **Normalization**: RMSNorm, LayerNorm, Gemma-style fused operations
- **Activations**: SiLU, GELU with fused gating

## PPU Support

| Hardware | Architecture |
|--------------|-------------------|
| ZhenWu 810e | ppu001 |
| ZhenWu M890 | ppu0015 |

> **Note:** Not all features are supported across all compute capabilities.

## Getting Started

## Requirements

Before installing Flashinfer-for-SAIL v0.6.12, make sure the SAIL SDK and required runtime components are available in your environment.

- SAIL SDK v2.1.1 or later
- Python 3.12
- PyTorch-for-SAIL 2.10.0 or later

For supported operating systems, CUDA Wrapper versions, and the full dependency list, see the Flashinfer-for-SAIL User Guide.

### Installation

**Quickstart:**

### Option 1: Use the Docker Image (Recommended)

FlashInfer-for-SAIL is pre-installed in the matching SGLang-for-SAIL v0.5.13 Docker image, so no extra installation is required. Using this image is recommended because it avoids manual setup of the base runtime environment.

Once inside the container, skip to [Verify Installation](#verify-installation).

### Option 2: Install from PyPI

```bash
pip install flashinfer-python -i https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple
```

- **flashinfer-python**: Core package that compiles/downloads kernels on first use

**For faster initialization and offline usage**, install the optional packages:

```bash
pip install flashinfer-python -i https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple  --no-deps --force
```

### Verify Installation

```bash
flashinfer show-config
```

### Basic Usage

```python
import torch
import flashinfer

# Single decode attention
q = torch.randn(32, 128, device="cuda", dtype=torch.float16)  # [num_qo_heads, head_dim]
k = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)  # [kv_len, num_kv_heads, head_dim]
v = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)

output = flashinfer.single_decode_with_kv_cache(q, k, v)
```

See [documentation](https://docs.flashinfer.ai/) for comprehensive API reference and tutorials.

### Install from Source

```bash
git clone https://github.com/flytiger-eco/flashinfer-for-sail.git -b v0.6.12 --recursive
cd flashinfer-for-sail
python -m pip install -v .
```

**For development**, install in editable mode:

```bash
python -m pip install --no-build-isolation -e . -v
```

> **Note:** When using `--no-build-isolation`, pip does not automatically install build dependencies. FlashInfer requires `setuptools>=77`. If you encounter an error like `AttributeError: module 'setuptools.build_meta' has no attribute 'prepare_metadata_for_build_editable'`, upgrade pip and setuptools first:
> ```bash
> python -m pip install --upgrade pip setuptools
> ```

Build optional packages:

```bash
# flashinfer-cubin
cd flashinfer-cubin
python -m build --no-isolation --wheel
python -m pip install dist/*.whl
```

```bash
# flashinfer-jit-cache (customize for your target PPUs)
export FLASHINFER_CUDA_ARCH_LIST="8.0 8.9"
cd flashinfer-jit-cache
python -m build --no-isolation --wheel
python -m pip install dist/*.whl
```

For more details, see the [Install from Source documentation](https://docs.flashinfer.ai/installation.html#install-from-source).

### CLI Tools

FlashInfer provides several CLI commands for configuration, module management, and development:

```bash
# Verify installation and view configuration
flashinfer show-config

# List and inspect modules
flashinfer list-modules
flashinfer module-status

# Manage artifacts and cache
flashinfer download-cubin
flashinfer clear-cache

# For developers: generate compile_commands.json for IDE integration
flashinfer export-compile-commands [output_path]
```

For complete documentation, see the [CLI reference](https://docs.flashinfer.ai/cli.html).

## API Logging

FlashInfer provides comprehensive API logging for debugging. Enable it using environment variables:

```bash
# Enable logging (levels: 0=off (default), 1=basic, 3=detailed, 5=statistics)
export FLASHINFER_LOGLEVEL=3

# Set log destination (stdout (default), stderr, or file path)
export FLASHINFER_LOGDEST=stdout
```

For detailed information about logging levels, configuration, and advanced features, see [Logging](https://docs.flashinfer.ai/logging.html) in our documentation.

## Custom Attention Variants

Users can customize their own attention variants with additional parameters. For more details, refer to our [JIT examples](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/utils/test_jit_example.py).

## Acknowledgement

FlashInfer is inspired by [FlashAttention](https://github.com/dao-AILab/flash-attention/), [vLLM](https://github.com/vllm-project/vllm), [stream-K](https://arxiv.org/abs/2301.03598), [CUTLASS](https://github.com/nvidia/cutlass), and [AITemplate](https://github.com/facebookincubator/AITemplate).

## Citation

If you find FlashInfer helpful in your project or research, please consider citing our [paper](https://arxiv.org/abs/2501.01005):

```bibtex
@article{ye2025flashinfer,
    title = {FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving},
    author = {
      Ye, Zihao and
      Chen, Lequn and
      Lai, Ruihang and
      Lin, Wuwei and
      Zhang, Yineng and
      Wang, Stephanie and
      Chen, Tianqi and
      Kasikci, Baris and
      Grover, Vinod and
      Krishnamurthy, Arvind and
      Ceze, Luis
    },
    journal = {arXiv preprint arXiv:2501.01005},
    year = {2025},
    url = {https://arxiv.org/abs/2501.01005}
}
```
