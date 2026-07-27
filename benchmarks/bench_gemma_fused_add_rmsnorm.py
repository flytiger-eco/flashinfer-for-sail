import argparse

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(range(1024, 16385, 1024)))
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=list(range(1024, 16385, 1024)),
    )
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float16", "bfloat16"], default=["float16"]
    )
    args = parser.parse_args()

    eps = 1e-6

    for batch_size in args.batch_sizes:
        for hidden_size in args.hidden_sizes:
            for dtype_str in args.dtypes:
                dtype = getattr(torch, dtype_str)

                x = torch.randn((batch_size, hidden_size), dtype=dtype, device="cuda")
                residual = torch.randn_like(x)
                weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

                @torch.cuda.nvtx.range(
                    f"gemma_fused_add_rmsnorm batch_size={batch_size}, hidden_size={hidden_size}, dtype={dtype_str}"
                )
                def fn() -> None:
                    flashinfer.gemma_fused_add_rmsnorm(x, residual, weight, eps)

                measurements = bench_gpu_time(fn)
                latency_ms = np.median(measurements)
                throughput = (
                    x.numel() * x.element_size() * 2
                    + residual.numel() * residual.element_size() * 2
                    + weight.numel() * weight.element_size()
                ) / (latency_ms * 1e-3)
                print(
                    f"batch_size: {batch_size:3},",
                    f"hidden_size: {hidden_size:5},",
                    f"dtype: {dtype_str:8},",
                    f"latency: {latency_ms * 1e3:2.0f}us,",
                    f"throughput: {throughput * 1e-9:7.3f}GB/s",
                )

        print("---")

    torch.cuda.profiler.stop()


if __name__ == "__main__":
    main()
