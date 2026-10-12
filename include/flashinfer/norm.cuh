/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_NORM_CUH_
#define FLASHINFER_NORM_CUH_

#include <cstdint>
#include <numeric>

#include "flashinfer/trtllm/common/cudaTypeUtils.cuh"
#include "flashinfer/trtllm/common/cudaUtils.h"
#include "flashinfer/trtllm/common/reduceKernelUtils.cuh"
#include "flashinfer/utils.cuh"
#include "math.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

namespace norm {

using namespace tensorrt_llm::common;

template <uint32_t VEC_SIZE, typename T, bool HAS_WEIGHT = true>
__global__ void RMSNormKernel(T* __restrict__ input, T* __restrict__ weight, T* __restrict__ output,
                              const uint32_t d, const uint32_t stride_input,
                              const uint32_t stride_output, float weight_bias, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  // NOTE(Zihao): it's guaranteed that num_warps should be smaller than 32
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];

  float sum_sq = 0.f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      sum_sq += float(input_vec[j]) * float(input_vec[j]);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += math::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    vec_t<T, VEC_SIZE> weight_vec;
    vec_t<T, VEC_SIZE> output_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      if constexpr (HAS_WEIGHT) {
        weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      }
    }
    if constexpr (HAS_WEIGHT) {
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        output_vec[j] = float(input_vec[j]) * rms_rcp * (weight_bias + float(weight_vec[j]));
      }
    } else {
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        output_vec[j] = float(input_vec[j]) * rms_rcp;
      }
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.store(output + bx * stride_output + i * num_threads * VEC_SIZE +
                       thread_id * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// RMSNormPersistentKernel: persistent double-buffered RMS normalization.
//
// Designed for decode workloads (small batch_size, large d) where we want each
// block to process multiple rows in a persistent grid fashion.
//
// Warp split:
//   Main group  (ty < num_warps_main): load input, reduce sum_sq, normalize output
//   Helper group (ty >= num_warps_main, NUM_HELPER_WARPS warps): cp.async prefetch +
//                                                                  cross-warp reduce
//
// Shared memory layout:
//   [float region, 16-byte aligned]
//     smem_warp_sums[num_warps_main] : per-warp sum_sq for current row
//     smem_rms_rcp[1]                : rms_rcp for current row
//   [T region after alignment]
//     smem_weight[d]                : weight row (prefetched once in prologue)
//     smem_input_next[d]            : next row input (double buffer)
//
// Execution flow:
//   Prologue: helper prefetches weight + row 1 (next row); main processes row 0 from global
//   Main loop: for each row, pipeline across 3 stages
//     Stage 0: Main: warp-reduces current row; loads next row data from smem to regs
//              [DEFER_WAIT=true]: also waits for previous iteration's cp.async
//     Stage 1: helper warp 0 cross-reduces → rms_rcp; main computes sum_sq_next from input_next;
//              Main prefetches next-next row via cp.async
//              [DEFER_WAIT=false]: commit + wait_group 0 together
//              [DEFER_WAIT=true]: commit only, wait deferred to next Stage 0
//     Stage 2: main normalizes current row using smem_weight → output; swap cur←next
template <uint32_t VEC_SIZE, typename T, uint32_t NUM_HELPER_WARPS = 4, bool DEFER_WAIT = false,
          bool HAS_WEIGHT = true>
__global__ void RMSNormPersistentKernel(
    T* __restrict__ input, T* __restrict__ weight, T* __restrict__ output,
    const uint32_t batch_size, const uint32_t d,
    const uint32_t stride_input, const uint32_t stride_output,
    float weight_bias, float eps) {
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps_total = blockDim.y;
  const uint32_t num_warps_main = num_warps_total - NUM_HELPER_WARPS;
  const bool is_helper = (ty >= num_warps_main);
  const uint32_t helper_ty = ty - num_warps_main;
  const uint32_t num_threads_main = num_warps_main * warp_size;
  const uint32_t main_thread_id = tx + ty * warp_size;
  const uint32_t helper_thread_id = tx + helper_ty * warp_size;
  const uint32_t num_helper_threads = NUM_HELPER_WARPS * warp_size;

  // rounds must be 1: each thread covers exactly one VEC_SIZE chunk
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads_main);
  constexpr uint32_t MAX_ROUNDS = 1;
  assert(rounds <= MAX_ROUNDS);

  // Shared memory layout
  extern __shared__ char smem_raw[];
  const uint32_t float_region_bytes = (num_warps_main + 1) * sizeof(float);
  const uint32_t aligned_float_region = (float_region_bytes + 15u) & ~15u;
  float* smem_warp_sums = reinterpret_cast<float*>(smem_raw);
  float* smem_rms_rcp   = smem_warp_sums + num_warps_main;
  T* smem_weight        = reinterpret_cast<T*>(smem_raw + aligned_float_region);
  T* smem_input_next    = smem_weight + d;

  constexpr uint32_t cp_size = VEC_SIZE * sizeof(T);

  // Per-thread registers for current row (main group only)
  float x_cur[VEC_SIZE];
  float sum_sq = 0.f;
  // Per-thread registers for next row (loaded from smem in Stage 0)
  float x_next[VEC_SIZE];
  float sum_sq_next = 0.f;

  // vec index for this thread (used throughout; 1 round only)
  const uint32_t vec_idx = main_thread_id * VEC_SIZE;
  const bool valid = vec_idx < d;

  // -------------------------------------------------------------------------
  // PDL: wait for grid dependencies
  // -------------------------------------------------------------------------
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // =========================================================================
  // Prologue (executed once before the main loop)
  // =========================================================================

  // --- Prologue Step 1: Helper prefetches weight → smem_weight ---
  // Skip prefetch if weight is nullptr; smem layout is preserved.
  if constexpr (HAS_WEIGHT) {
    if (is_helper) {
      const uint32_t copy_rounds_w = ceil_div(d, VEC_SIZE * num_helper_threads);
      for (uint32_t i = 0; i < copy_rounds_w; i++) {
        const uint32_t e = (i * num_helper_threads + helper_thread_id) * VEC_SIZE;
        if (e < d) {
          if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
            uint32_t dst_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(smem_weight + e));
            const T* src_ptr = weight + e;
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                         ::"r"(dst_addr), "l"(src_ptr), "n"(cp_size));
          } else {
            vec_t<T, VEC_SIZE> tmp;
            tmp.fill(0.f);
            tmp.load(weight + e);
            tmp.store(smem_weight + e);
          }
        }
      }
    }
  }

  // --- Prologue Step 2: Main loads row 0 from global, computes x_cur ---
  if (!is_helper) {
    vec_t<T, VEC_SIZE> inp_vec;
    inp_vec.fill(0.f);
    if (valid && blockIdx.x < batch_size) {
      inp_vec.load(input + blockIdx.x * stride_input + vec_idx);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float xi = float(inp_vec[j]);
      x_cur[j]  = xi;
      sum_sq   += xi * xi;
    }
  }

  // --- Prologue Step 3: Helper prefetches row 1 (blockIdx.x + gridDim.x) → smem double buffer ---
  if (is_helper) {
    const uint32_t next_row = blockIdx.x + gridDim.x;
    if (next_row < batch_size) {
      const uint32_t copy_rounds_r = ceil_div(d, VEC_SIZE * num_helper_threads);
      for (uint32_t i = 0; i < copy_rounds_r; i++) {
        const uint32_t e = (i * num_helper_threads + helper_thread_id) * VEC_SIZE;
        if (e < d) {
          if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
            uint32_t dst_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(smem_input_next + e));
            const T* src_inp = input + next_row * stride_input + e;
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                         ::"r"(dst_addr), "l"(src_inp), "n"(cp_size));
          } else {
            vec_t<T, VEC_SIZE> tmp;
            tmp.fill(0.f);
            tmp.load(input + next_row * stride_input + e);
            tmp.store(smem_input_next + e);
          }
        }
      }
    }
    if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
      asm volatile("cp.async.commit_group;\n");
      asm volatile("cp.async.wait_group 0;\n");
    }
  }
  __syncthreads();  // Prologue complete: weight + row1 data visible to all threads

  // =========================================================================
  // Main loop: persistent, step = gridDim.x
  // =========================================================================
  for (uint32_t row = blockIdx.x; row < batch_size; row += gridDim.x) {
    const bool has_next = (row + gridDim.x) < batch_size;
    const bool has_next_next = (row + 2 * gridDim.x) < batch_size;

    // -----------------------------------------------------------------
    // Stage 0:
    //   Main: wait for previous iteration's cp.async;
    //         warp-reduce sum_sq → smem_warp_sums[ty]
    //         if has_next: load next row from smem → regs (inp_next_vec)
    //   Helper: idle
    // -----------------------------------------------------------------
    vec_t<T, VEC_SIZE> inp_next_vec;
    inp_next_vec.fill(0.f);

    if (!is_helper) {
      // Wait for previous iteration's cp.async (only in defer mode)
      if constexpr (DEFER_WAIT) {
        if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
          asm volatile("cp.async.wait_group 0;\n");
        }
      }

      // Warp-level butterfly reduce
#pragma unroll
      for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
        sum_sq += math::shfl_xor_sync(sum_sq, offset);
      }
      if (tx == 0) {
        smem_warp_sums[ty] = sum_sq;
      }
      // Load next row from smem double buffer to registers
      if (has_next && valid) {
        inp_next_vec.load(smem_input_next + vec_idx);
      }
    }
    __syncthreads();  // Stage 0 → Stage 1

    // -----------------------------------------------------------------
    // Stage 1:
    //   Helper warp 0: cross-reduce smem_warp_sums → smem_rms_rcp[0]
    //   Main: if has_next, compute x_next, sum_sq_next;
    //         Main prefetches next-next row via cp.async
    //   Helper warps 1-3: idle
    // -----------------------------------------------------------------
    sum_sq_next = 0.f;
    if (is_helper) {
      if (helper_ty == 0) {
        float val = (tx < num_warps_main) ? smem_warp_sums[tx] : 0.f;
#pragma unroll
        for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
          val += math::shfl_xor_sync(val, offset);
        }
        if (tx == 0) {
          smem_rms_rcp[0] = math::rsqrt(val / float(d) + eps);
        }
      }
    } else {
      // Main: compute x_next, accumulate sum_sq_next
      if (has_next) {
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
          float xi = float(inp_next_vec[j]);
          x_next[j]    = xi;
          sum_sq_next += xi * xi;
        }
      }

      // Main: issue cp.async for next-next row
      if (has_next_next) {
        const uint32_t next_next_row = row + 2 * gridDim.x;
        const uint32_t copy_rounds_p = ceil_div(d, VEC_SIZE * num_threads_main);
        for (uint32_t i = 0; i < copy_rounds_p; i++) {
          const uint32_t e = (i * num_threads_main + main_thread_id) * VEC_SIZE;
          if (e < d) {
            if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
              uint32_t dst_addr = static_cast<uint32_t>(
                  __cvta_generic_to_shared(smem_input_next + e));
              const T* src_inp = input + next_next_row * stride_input + e;
              asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                           ::"r"(dst_addr), "l"(src_inp), "n"(cp_size));
            } else {
              vec_t<T, VEC_SIZE> tmp;
              tmp.fill(0.f);
              tmp.load(input + next_next_row * stride_input + e);
              tmp.store(smem_input_next + e);
            }
          }
        }
      }
      if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
        asm volatile("cp.async.commit_group;\n");
        if constexpr (!DEFER_WAIT) {
          asm volatile("cp.async.wait_group 0;\n");
        }
      }
    }
    __syncthreads();  // Stage 1 → Stage 2

    // -----------------------------------------------------------------
    // Stage 2:
    //   Main: read smem_rms_rcp[0], load weight from smem_weight,
    //         normalize x_cur, write to output[row]
    //   Swap: x_cur ← x_next, sum_sq ← sum_sq_next
    // -----------------------------------------------------------------
    if (!is_helper) {
      const float rms_rcp = smem_rms_rcp[0];
      if (valid && row < batch_size) {
        vec_t<T, VEC_SIZE> weight_vec, out_vec;
        weight_vec.fill(0.f);
        out_vec.fill(0.f);
        if constexpr (HAS_WEIGHT) {
          weight_vec.load(smem_weight + vec_idx);
#pragma unroll
          for (uint32_t j = 0; j < VEC_SIZE; j++) {
            out_vec[j] = x_cur[j] * rms_rcp * (weight_bias + float(weight_vec[j]));
          }
        } else {
#pragma unroll
          for (uint32_t j = 0; j < VEC_SIZE; j++) {
            out_vec[j] = x_cur[j] * rms_rcp;
          }
        }
        out_vec.store(output + row * stride_output + vec_idx);
      }
      // Swap current ← next (for last iteration, this is a no-op logically)
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        x_cur[j] = x_next[j];
      }
      sum_sq = sum_sq_next;
    }
    // No __syncthreads() needed here: next iteration's Stage 0 smem write
    // (smem_warp_sums) happens before Stage 1 reads it, protected by the
    // __syncthreads() at the end of Stage 0.
  }  // end persistent loop

  // =========================================================================
  // PDL: signal dependents
  // =========================================================================
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}  // end RMSNormPersistentKernel

template <typename T>
cudaError_t RMSNorm(T* input, T* weight, T* output, uint32_t batch_size, uint32_t d,
                    uint32_t stride_input, uint32_t stride_output, float eps = 1e-5,
                    bool enable_pdl = false, cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32);
  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = num_warps * sizeof(float);
  float weight_bias = 0.f;
  void* args[] = {&input, &weight, &output, &d, &stride_input, &stride_output, &weight_bias, &eps};

  cudaLaunchConfig_t config;
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {

    constexpr uint32_t NUM_HELPER_WARPS = 4;
    const uint32_t num_threads_main = num_warps * 32;
    const uint32_t rounds = ceil_div(d, (uint32_t)VEC_SIZE * num_threads_main);

    // --- Persistent kernel path (highest priority) ---
    bool used_persistent = false;
    if (num_warps + NUM_HELPER_WARPS <= 32 && rounds <= 1) {
      const uint32_t num_warps_total = num_warps + NUM_HELPER_WARPS;
      // Compute persistent kernel smem size:
      //   float region: (num_warps + 1) floats for warp_sums + rms_rcp, 16-byte aligned
      //   T region: 2 * d elements (weight + input_next)
      const uint32_t float_region_bytes =
          (num_warps + 1u) * static_cast<uint32_t>(sizeof(float));
      const uint32_t aligned_float_region = (float_region_bytes + 15u) & ~15u;
      const uint32_t persistent_smem_size =
          aligned_float_region + 2u * d * static_cast<uint32_t>(sizeof(T));

      auto persistent_kernel_fn =
          RMSNormPersistentKernel<VEC_SIZE, T, NUM_HELPER_WARPS, false, true>;

      int num_blocks_per_sm = 0, num_sms = 0, dev_id = 0;
      // cudaFuncSetAttribute must come before cudaOccupancyMaxActiveBlocksPerMultiprocessor
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(persistent_kernel_fn,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               persistent_smem_size));
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm, persistent_kernel_fn, num_warps_total * 32, persistent_smem_size));
      FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
      FLASHINFER_CUDA_CALL(
          cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

      const uint32_t total_persistent_blocks =
          static_cast<uint32_t>(num_blocks_per_sm * num_sms);

      if (total_persistent_blocks > 0) {
        const uint32_t num_persistent_blocks =
            std::min(batch_size, total_persistent_blocks);
        dim3 nblks_persistent(num_persistent_blocks);
        dim3 nthrs_persistent(32, num_warps_total);
        config.gridDim = nblks_persistent;
        config.blockDim = nthrs_persistent;
        config.dynamicSmemBytes = persistent_smem_size;

        if (batch_size > 2 * num_persistent_blocks) {
          // Large bs: defer wait to next iteration for better overlap
          if (weight != nullptr) {
            auto kernel_fn =
                RMSNormPersistentKernel<VEC_SIZE, T, NUM_HELPER_WARPS, true, true>;
            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kernel_fn,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     persistent_smem_size));
            FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel_fn, input, weight,
                                                   output, batch_size, d, stride_input,
                                                   stride_output, weight_bias, eps));
          } else {
            auto kernel_fn =
                RMSNormPersistentKernel<VEC_SIZE, T, NUM_HELPER_WARPS, true, false>;
            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kernel_fn,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     persistent_smem_size));
            FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel_fn, input, weight,
                                                   output, batch_size, d, stride_input,
                                                   stride_output, weight_bias, eps));
          }
        } else {
          // Small bs: commit + wait together in Stage 1
          if (weight != nullptr) {
            FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, persistent_kernel_fn, input, weight,
                                                   output, batch_size, d, stride_input,
                                                   stride_output, weight_bias, eps));
          } else {
            auto kernel_fn =
                RMSNormPersistentKernel<VEC_SIZE, T, NUM_HELPER_WARPS, false, false>;
            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kernel_fn,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     persistent_smem_size));
            FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel_fn, input, weight,
                                                   output, batch_size, d, stride_input,
                                                   stride_output, weight_bias, eps));
          }
        }
        used_persistent = true;
      }
    }

    if (!used_persistent) {
      // Fallback: original single-row kernel, one block processes one row
      dim3 nblks(batch_size);
      dim3 nthrs(32, num_warps);
      const uint32_t smem_size = num_warps * sizeof(float);
      config.gridDim = nblks;
      config.blockDim = nthrs;
      config.dynamicSmemBytes = smem_size;

      if (weight != nullptr) {
        auto kernel_fn = RMSNormKernel<VEC_SIZE, T, true>;
        FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel_fn, input, weight, output, d,
                                                stride_input, stride_output, weight_bias, eps));
      } else {
        auto kernel_fn = RMSNormKernel<VEC_SIZE, T, false>;
        FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(
            kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel_fn, input, weight, output, d,
                                                stride_input, stride_output, weight_bias, eps));
      }
    }
  });
  return cudaSuccess;
}

template <uint32_t VEC_SIZE, typename T, typename O>
__global__ void RMSNormQuantKernel(T* __restrict__ input, T* __restrict__ weight,
                                   O* __restrict__ output, const uint32_t d,
                                   const uint32_t stride_input, const uint32_t stride_output,
                                   float weight_bias, float* scale, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  // NOTE(Zihao): it's guaranteed that num_warps should be smaller than 32
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  const float scale_inv = 1.0f / scale[0];
  extern __shared__ float smem[];

  float sum_sq = 0.f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      sum_sq += float(input_vec[j]) * float(input_vec[j]);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += math::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    vec_t<T, VEC_SIZE> weight_vec;
    vec_t<float, VEC_SIZE> output_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] =
          float(input_vec[j]) * rms_rcp * (weight_bias + float(weight_vec[j])) * scale_inv;
      output_vec[j] = fmaxf(-448.0f, fminf(output_vec[j], 448.0f));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.cast_store(output + bx * stride_output + i * num_threads * VEC_SIZE +
                            thread_id * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <uint32_t VEC_SIZE, typename T, typename O>
__global__ void RMSNormOnlineQuantKernel(
    T* __restrict__ input, T* __restrict__ weight,
    O* __restrict__ output, float* __restrict__ output_scale,
    const uint32_t d, const uint32_t stride_input, const uint32_t stride_output,
    float weight_bias, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  constexpr float FP8_E4M3_MAX = 448.0f;
  extern __shared__ float smem[];

  float sum_sq = 0.f;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // =========================================================================
  // Pass 1: compute sum_sq -> rms_rcp
  // =========================================================================
  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      sum_sq += float(input_vec[j]) * float(input_vec[j]);
    }
  }

  // warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += math::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // cross warp reduce sum
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);

  // =========================================================================
  // Pass 2: compute normalized values, find per-token absmax -> scale
  // =========================================================================
  float max_abs = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    vec_t<T, VEC_SIZE> weight_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float norm_val = float(input_vec[j]) * rms_rcp * (weight_bias + float(weight_vec[j]));
      max_abs = fmaxf(max_abs, fabsf(norm_val));
    }
  }

  // warp reduce max
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, math::shfl_xor_sync(max_abs, offset));
  }

  smem[ty] = max_abs;
  __syncthreads();
  // cross warp reduce max
  if (ty == 0) {
    max_abs = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      max_abs = fmaxf(max_abs, math::shfl_xor_sync(max_abs, offset));
    }
    smem[0] = max_abs;
  }
  __syncthreads();

  max_abs = smem[0];
  const float scale = max_abs / FP8_E4M3_MAX;
  const float scale_inv = (max_abs > 0.f) ? FP8_E4M3_MAX / max_abs : 0.f;

  // Write per-token scale
  if (ty == 0 && tx == 0) {
    output_scale[bx] = scale;
  }

  // =========================================================================
  // Pass 3: quantize and write output
  // =========================================================================
  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    vec_t<T, VEC_SIZE> weight_vec;
    vec_t<float, VEC_SIZE> output_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float norm_val = float(input_vec[j]) * rms_rcp * (weight_bias + float(weight_vec[j]));
      output_vec[j] = fmaxf(-FP8_E4M3_MAX, fminf(norm_val * scale_inv, FP8_E4M3_MAX));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.cast_store(output + bx * stride_output + i * num_threads * VEC_SIZE +
                            thread_id * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// RMSNormOnlineQuantPersistentKernel: persistent double-buffered RMS normalization with online
// FP8 quantization.
//
// Similar to RMSNormPersistentKernel but requires 2 reductions per row:
//   1. sum_sq → rms_rcp
//   2. max_abs(norm_val) → scale_inv
// Then quantizes output using scale_inv.
//
// 5 stages per row:
//   Stage 0: Main: warp-reduces sum_sq; loads next row from smem → regs
//            [DEFER_WAIT=true]: also waits for previous iteration's cp.async
//   Stage 1: Helper warp 0 cross-reduces sum_sq → rms_rcp;
//            Main computes x_next/sum_sq_next;
//            Main prefetches next-next row via cp.async
//            [DEFER_WAIT=false]: commit + wait_group 0 together
//            [DEFER_WAIT=true]: commit only, wait deferred to next Stage 0
//   Stage 2: Main normalizes x_cur, computes max_abs, warp-reduces max_abs
//   Stage 3: Helper warp 0 cross-reduces max_abs → scale_inv; writes output_scale
//   Stage 4: Main quantizes x_cur → output; swaps cur←next
template <uint32_t VEC_SIZE, typename T, typename O, uint32_t NUM_HELPER_WARPS = 4, bool DEFER_WAIT = false>
__global__ void RMSNormOnlineQuantPersistentKernel(
    T* __restrict__ input, T* __restrict__ weight,
    O* __restrict__ output, float* __restrict__ output_scale,
    const uint32_t batch_size, const uint32_t d,
    const uint32_t stride_input, const uint32_t stride_output,
    float weight_bias, float eps) {
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps_total = blockDim.y;
  const uint32_t num_warps_main = num_warps_total - NUM_HELPER_WARPS;
  const bool is_helper = (ty >= num_warps_main);
  const uint32_t helper_ty = ty - num_warps_main;
  const uint32_t num_threads_main = num_warps_main * warp_size;
  const uint32_t main_thread_id = tx + ty * warp_size;
  const uint32_t helper_thread_id = tx + helper_ty * warp_size;
  const uint32_t num_helper_threads = NUM_HELPER_WARPS * warp_size;

  constexpr float FP8_E4M3_MAX = 448.0f;

  // rounds must be 1: each thread covers exactly one VEC_SIZE chunk
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads_main);
  constexpr uint32_t MAX_ROUNDS = 1;
  assert(rounds <= MAX_ROUNDS);

  // Shared memory layout
  extern __shared__ char smem_raw[];
  const uint32_t float_region_bytes = (num_warps_main + 1) * sizeof(float);
  const uint32_t aligned_float_region = (float_region_bytes + 15u) & ~15u;
  float* smem_warp_sums = reinterpret_cast<float*>(smem_raw);
  float* smem_rms_rcp   = smem_warp_sums + num_warps_main;
  T* smem_weight        = reinterpret_cast<T*>(smem_raw + aligned_float_region);
  T* smem_input_next    = smem_weight + d;

  constexpr uint32_t cp_size = VEC_SIZE * sizeof(T);

  // Per-thread registers
  float x_cur[VEC_SIZE];
  float sum_sq = 0.f;
  float x_next[VEC_SIZE];
  float sum_sq_next = 0.f;
  float max_abs = 0.f;

  // vec index for this thread (1 round only)
  const uint32_t vec_idx = main_thread_id * VEC_SIZE;
  const bool valid = vec_idx < d;

  // -------------------------------------------------------------------------
  // PDL: wait for grid dependencies
  // -------------------------------------------------------------------------
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // =========================================================================
  // Prologue
  // =========================================================================

  // --- Prologue Step 1: Helper prefetches weight → smem_weight ---
  if (is_helper) {
    const uint32_t copy_rounds_w = ceil_div(d, VEC_SIZE * num_helper_threads);
    for (uint32_t i = 0; i < copy_rounds_w; i++) {
      const uint32_t e = (i * num_helper_threads + helper_thread_id) * VEC_SIZE;
      if (e < d) {
        if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
          uint32_t dst_addr = static_cast<uint32_t>(
              __cvta_generic_to_shared(smem_weight + e));
          const T* src_ptr = weight + e;
          asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                       ::"r"(dst_addr), "l"(src_ptr), "n"(cp_size));
        } else {
          vec_t<T, VEC_SIZE> tmp;
          tmp.fill(0.f);
          tmp.load(weight + e);
          tmp.store(smem_weight + e);
        }
      }
    }
  }

  // --- Prologue Step 2: Main loads row 0 from global, computes x_cur ---
  if (!is_helper) {
    vec_t<T, VEC_SIZE> inp_vec;
    inp_vec.fill(0.f);
    if (valid && blockIdx.x < batch_size) {
      inp_vec.load(input + blockIdx.x * stride_input + vec_idx);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float xi = float(inp_vec[j]);
      x_cur[j]  = xi;
      sum_sq   += xi * xi;
    }
  }

  // --- Prologue Step 3: Helper prefetches row 1 → smem_input_next ---
  if (is_helper) {
    const uint32_t next_row = blockIdx.x + gridDim.x;
    if (next_row < batch_size) {
      const uint32_t copy_rounds_r = ceil_div(d, VEC_SIZE * num_helper_threads);
      for (uint32_t i = 0; i < copy_rounds_r; i++) {
        const uint32_t e = (i * num_helper_threads + helper_thread_id) * VEC_SIZE;
        if (e < d) {
          if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
            uint32_t dst_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(smem_input_next + e));
            const T* src_inp = input + next_row * stride_input + e;
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                         ::"r"(dst_addr), "l"(src_inp), "n"(cp_size));
          } else {
            vec_t<T, VEC_SIZE> tmp;
            tmp.fill(0.f);
            tmp.load(input + next_row * stride_input + e);
            tmp.store(smem_input_next + e);
          }
        }
      }
    }
    if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
      asm volatile("cp.async.commit_group;\n");
      asm volatile("cp.async.wait_group 0;\n");
    }
  }
  __syncthreads();  // Prologue complete

  // =========================================================================
  // Main loop: persistent, step = gridDim.x
  // =========================================================================
  for (uint32_t row = blockIdx.x; row < batch_size; row += gridDim.x) {
    const bool has_next = (row + gridDim.x) < batch_size;
    const bool has_next_next = (row + 2 * gridDim.x) < batch_size;

    // -----------------------------------------------------------------
    // Stage 0: Main: warp-reduce sum_sq → smem_warp_sums[ty]
    //          if has_next: load next row from smem → regs
    //          [DEFER_WAIT=true]: wait for previous iteration's cp.async
    //          Helper: idle
    // -----------------------------------------------------------------
    vec_t<T, VEC_SIZE> inp_next_vec;
    inp_next_vec.fill(0.f);

    if (!is_helper) {
      // Wait for previous iteration's cp.async (only in defer mode)
      if constexpr (DEFER_WAIT) {
        if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
          asm volatile("cp.async.wait_group 0;\n");
        }
      }

#pragma unroll
      for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
        sum_sq += math::shfl_xor_sync(sum_sq, offset);
      }
      if (tx == 0) {
        smem_warp_sums[ty] = sum_sq;
      }
      if (has_next && valid) {
        inp_next_vec.load(smem_input_next + vec_idx);
      }
    }
    __syncthreads();  // Stage 0 → Stage 1

    // -----------------------------------------------------------------
    // Stage 1: Helper warp 0 cross-reduces sum_sq → smem_rms_rcp[0]
    //          Main: compute x_next, sum_sq_next;
    //                Main prefetches next-next row (cp.async, wait deferred to next Stage 0)
    //          Helper warps 1-3: idle
    // -----------------------------------------------------------------
    sum_sq_next = 0.f;
    if (is_helper) {
      if (helper_ty == 0) {
        float val = (tx < num_warps_main) ? smem_warp_sums[tx] : 0.f;
#pragma unroll
        for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
          val += math::shfl_xor_sync(val, offset);
        }
        if (tx == 0) {
          smem_rms_rcp[0] = math::rsqrt(val / float(d) + eps);
        }
      }
    } else {
      // Main: compute x_next, accumulate sum_sq_next
      if (has_next) {
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
          float xi = float(inp_next_vec[j]);
          x_next[j]    = xi;
          sum_sq_next += xi * xi;
        }
      }

      // Main: issue cp.async for next-next row
      if (has_next_next) {
        const uint32_t next_next_row = row + 2 * gridDim.x;
        const uint32_t copy_rounds_p = ceil_div(d, VEC_SIZE * num_threads_main);
        for (uint32_t i = 0; i < copy_rounds_p; i++) {
          const uint32_t e = (i * num_threads_main + main_thread_id) * VEC_SIZE;
          if (e < d) {
            if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
              uint32_t dst_addr = static_cast<uint32_t>(
                  __cvta_generic_to_shared(smem_input_next + e));
              const T* src_inp = input + next_next_row * stride_input + e;
              asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                           ::"r"(dst_addr), "l"(src_inp), "n"(cp_size));
            } else {
              vec_t<T, VEC_SIZE> tmp;
              tmp.fill(0.f);
              tmp.load(input + next_next_row * stride_input + e);
              tmp.store(smem_input_next + e);
            }
          }
        }
      }
      if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
        asm volatile("cp.async.commit_group;\n");
        if constexpr (!DEFER_WAIT) {
          asm volatile("cp.async.wait_group 0;\n");
        }
      }
    }
    __syncthreads();  // Stage 1 → Stage 2

    // -----------------------------------------------------------------
    // Stage 2: Main reads rms_rcp, loads weight, normalizes x_cur,
    //          computes max_abs, warp-reduces max_abs
    //          Helper: idle
    // -----------------------------------------------------------------
    max_abs = 0.f;
    if (!is_helper) {
      const float rms_rcp = smem_rms_rcp[0];
      if (valid && row < batch_size) {
        vec_t<T, VEC_SIZE> weight_vec;
        weight_vec.fill(0.f);
        weight_vec.load(smem_weight + vec_idx);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
          x_cur[j] = x_cur[j] * rms_rcp * (weight_bias + float(weight_vec[j]));
          max_abs = fmaxf(max_abs, fabsf(x_cur[j]));
        }
      }
      // Warp-level max reduce
#pragma unroll
      for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
        max_abs = fmaxf(max_abs, math::shfl_xor_sync(max_abs, offset));
      }
      if (tx == 0) {
        smem_warp_sums[ty] = max_abs;
      }
    }
    __syncthreads();  // Stage 2 → Stage 3

    // -----------------------------------------------------------------
    // Stage 3: Helper warp 0 cross-reduces max_abs → scale_inv
    //          → output_scale[row] = scale
    //          → smem_rms_rcp[0] = scale_inv
    //          Main + other helpers: idle
    // -----------------------------------------------------------------
    if (is_helper && helper_ty == 0) {
      float val = (tx < num_warps_main) ? smem_warp_sums[tx] : 0.f;
#pragma unroll
      for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, math::shfl_xor_sync(val, offset));
      }
      if (tx == 0) {
        const float scale = val / FP8_E4M3_MAX;
        const float scale_inv = (val > 0.f) ? FP8_E4M3_MAX / val : 0.f;
        smem_rms_rcp[0] = scale_inv;
        output_scale[row] = scale;
      }
    }
    __syncthreads();  // Stage 3 → Stage 4

    // -----------------------------------------------------------------
    // Stage 4: Main reads scale_inv, quantizes x_cur → output
    //          Swap: x_cur ← x_next, sum_sq ← sum_sq_next
    // -----------------------------------------------------------------
    if (!is_helper) {
      const float scale_inv = smem_rms_rcp[0];
      if (valid && row < batch_size) {
        vec_t<float, VEC_SIZE> out_vec;
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
          out_vec[j] = fmaxf(-FP8_E4M3_MAX, fminf(x_cur[j] * scale_inv, FP8_E4M3_MAX));
        }
        out_vec.cast_store(output + row * stride_output + vec_idx);
      }
      // Swap current ← next
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        x_cur[j] = x_next[j];
      }
      sum_sq = sum_sq_next;
    }
    // No __syncthreads() at end — next iteration's Stage 0 smem write
    // (smem_warp_sums) happens before Stage 1 reads it, protected by the
    // __syncthreads() at the end of Stage 0.
  }  // end persistent loop

  // =========================================================================
  // PDL: signal dependents
  // =========================================================================
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}  // end RMSNormOnlineQuantPersistentKernel

template <typename T, typename O>
cudaError_t RMSNormOnlineQuant(T* input, T* weight, O* output, float* output_scale,
                                uint32_t batch_size, uint32_t d,
                                uint32_t stride_input, uint32_t stride_output,
                                float eps = 1e-5, bool enable_pdl = false,
                                cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32);
  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = num_warps * sizeof(float);
  float weight_bias = 0.f;

  cudaLaunchConfig_t config;
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {

    constexpr uint32_t NUM_HELPER_WARPS = 4;
    const uint32_t num_threads_main = num_warps * 32;
    const uint32_t rounds = ceil_div(d, (uint32_t)VEC_SIZE * num_threads_main);

    // --- Persistent kernel path (highest priority) ---
    bool used_persistent = false;
    if (num_warps + NUM_HELPER_WARPS <= 32 && rounds <= 1) {
      const uint32_t num_warps_total = num_warps + NUM_HELPER_WARPS;
      const uint32_t float_region_bytes =
          (num_warps + 1u) * static_cast<uint32_t>(sizeof(float));
      const uint32_t aligned_float_region = (float_region_bytes + 15u) & ~15u;
      const uint32_t persistent_smem_size =
          aligned_float_region + 2u * d * static_cast<uint32_t>(sizeof(T));

      auto persistent_kernel_fn =
          RMSNormOnlineQuantPersistentKernel<VEC_SIZE, T, O, NUM_HELPER_WARPS, false>;

      int num_blocks_per_sm = 0, num_sms = 0, dev_id = 0;
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(persistent_kernel_fn,
                                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               persistent_smem_size));
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm, persistent_kernel_fn, num_warps_total * 32, persistent_smem_size));
      FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
      FLASHINFER_CUDA_CALL(
          cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

      const uint32_t total_persistent_blocks =
          static_cast<uint32_t>(num_blocks_per_sm * num_sms);

      if (total_persistent_blocks > 0) {
        const uint32_t num_persistent_blocks =
            std::min(batch_size, total_persistent_blocks);
        dim3 nblks_persistent(num_persistent_blocks);
        dim3 nthrs_persistent(32, num_warps_total);
        config.gridDim = nblks_persistent;
        config.blockDim = nthrs_persistent;
        config.dynamicSmemBytes = persistent_smem_size;

        if (batch_size > 2 * num_persistent_blocks) {
          // Large bs: defer wait to next iteration for better overlap
          auto kernel_fn = RMSNormOnlineQuantPersistentKernel<VEC_SIZE, T, O, NUM_HELPER_WARPS, true>;
          FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kernel_fn,
                                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                   persistent_smem_size));
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel_fn, input, weight,
                                                 output, output_scale, batch_size, d, stride_input,
                                                 stride_output, weight_bias, eps));
        } else {
          // Small bs: commit + wait together in Stage 1
          FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, persistent_kernel_fn, input, weight,
                                                 output, output_scale, batch_size, d, stride_input,
                                                 stride_output, weight_bias, eps));
        }
        used_persistent = true;
      }
    }

    if (!used_persistent) {
      // Fallback: original single-row kernel, one block processes one row
      config.gridDim = nblks;
      config.blockDim = nthrs;
      config.dynamicSmemBytes = smem_size;

      auto kernel = RMSNormOnlineQuantKernel<VEC_SIZE, T, O>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, input, weight, output, output_scale,
                                              d, stride_input, stride_output, weight_bias, eps));
    }
  });
  return cudaSuccess;
}

template <typename T, typename O>
cudaError_t RMSNormQuant(T* input, T* weight, O* output, uint32_t batch_size, uint32_t d,
                         uint32_t stride_input, uint32_t stride_output, float* scale,
                         float eps = 1e-5, bool enable_pdl = false, cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32);
  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = num_warps * sizeof(float);
  float weight_bias = 0.f;

  cudaLaunchConfig_t config;
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = RMSNormQuantKernel<VEC_SIZE, T, O>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, input, weight, output, d, stride_input,
                                            stride_output, weight_bias, scale, eps));
  });
  return cudaSuccess;
}

template <uint32_t VEC_SIZE, typename T, bool HAS_WEIGHT = true>
__global__ void QKRMSNormKernel(T* __restrict__ input, T* __restrict__ weight,
                                T* __restrict__ output, const uint32_t d, const uint32_t batch_size,
                                const uint32_t num_heads, const uint32_t stride_input_n,
                                const uint32_t stride_input_h, const uint32_t stride_output_n,
                                const uint32_t stride_output_h, float weight_bias, float eps) {
  const uint32_t num_blks = gridDim.x, num_warps = blockDim.y;
  const uint32_t num_workers = num_blks * num_warps;  // unroll on warp-dim
  const uint32_t num_jobs = batch_size * num_heads;

    const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  const uint32_t worker_idx = bx * num_warps + ty;

  constexpr uint32_t warp_size = 32;
  const uint32_t num_threads = warp_size;
  const uint32_t thread_id = tx;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t job_idx = worker_idx; job_idx < num_jobs; job_idx += num_workers) {
    // clear buffer
    float sum_sq = 0.f;

    // map back to batch-idx and head-idx; layout [batch_size, num_heads, head_dim]
    const uint32_t batch_idx = job_idx / num_heads;
    const uint32_t head_idx = job_idx % num_heads;

    for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + batch_idx * stride_input_n + head_idx * stride_input_h +
                       i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
            sum_sq += float(input_vec[j]) * float(input_vec[j]);
      }
    }

    // only have warp reduce sum
    // no need for __syncwarps as shfl already sync
#pragma unroll
      for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
        sum_sq += math::shfl_xor_sync(sum_sq, offset);
      }
      
    float rms_rcp = math::rsqrt(sum_sq / float(d) + eps);

    for (uint32_t i = 0; i < rounds; i++) {
      vec_t<T, VEC_SIZE> input_vec;
      vec_t<T, VEC_SIZE> weight_vec;
      vec_t<T, VEC_SIZE> output_vec;
      input_vec.fill(0.f);
      weight_vec.fill(0.f);
      if ((i * num_threads + thread_id) * VEC_SIZE < d) {
            input_vec.load(input + batch_idx * stride_input_n + head_idx * stride_input_h +
                       i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
        if constexpr (HAS_WEIGHT) {
          weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
        }
      }
        if constexpr (HAS_WEIGHT) {
          #pragma unroll
          for (uint32_t j = 0; j < VEC_SIZE; j++) {
            output_vec[j] = float(input_vec[j]) * rms_rcp * (weight_bias + float(weight_vec[j]));
          }
        } else {
#pragma unroll
          for (uint32_t j = 0; j < VEC_SIZE; j++) {
            output_vec[j] = float(input_vec[j]) * rms_rcp;
          }
        }
if ((i * num_threads + thread_id) * VEC_SIZE < d) {
        output_vec.store(output + batch_idx * stride_output_n + head_idx * stride_output_h +
                      i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      }
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}
// QKRMSNormPersistentKernel:
//   Multi-head parallel persistent variant of QKRMSNormKernel for 3D
//   (batch, num_heads, d) input/output layout with separate strides per-batch
//   and per-head.
//
//   Design:
//     - Each main warp ty in [0, NUM_HEADS_PER_ITER) independently processes
//       one head per iteration: tx is the vec index, the warp reduces sum_sq
//       in-register via shfl_xor, eliminating the cross-warp reduction.
//     - Helper warps (NUM_HELPER_WARPS) cooperatively prefetch a batch of
//       NUM_HEADS_PER_ITER heads into shared memory using a flat index.
//     - Each block consumes NUM_HEADS_PER_ITER rows per iteration; grid stride
//       is gridDim.x * NUM_HEADS_PER_ITER.
//
//   num_jobs = batch_size * num_heads. Each row index decomposes as
//     batch_idx = row / num_heads
//     head_idx  = row % num_heads
//   and is mapped to addresses:
//     input  + batch_idx * stride_input_n  + head_idx * stride_input_h  + offset
//     output + batch_idx * stride_output_n + head_idx * stride_output_h + offset
template <uint32_t VEC_SIZE, typename T, uint32_t NUM_HEADS_PER_ITER,
          uint32_t NUM_HELPER_WARPS = 4, bool DEFER_WAIT = false, bool HAS_WEIGHT = true>
__global__ void QKRMSNormPersistentKernel(
    T* __restrict__ input, T* __restrict__ weight, T* __restrict__ output,
    const uint32_t d, const uint32_t batch_size, const uint32_t num_heads,
    const uint32_t stride_input_n, const uint32_t stride_input_h,
    const uint32_t stride_output_n, const uint32_t stride_output_h,
    float weight_bias, float eps) {
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const bool is_helper = (ty >= NUM_HEADS_PER_ITER);
  const uint32_t helper_ty = ty - NUM_HEADS_PER_ITER;
  const uint32_t helper_thread_id = tx + helper_ty * warp_size;
  const uint32_t num_helper_threads = NUM_HELPER_WARPS * warp_size;

  const uint32_t num_jobs = batch_size * num_heads;
  const uint32_t grid_stride = gridDim.x * NUM_HEADS_PER_ITER;

  // Each main warp ty processes one head; tx is the vec index within that head.
  // Constraint: d <= warp_size * VEC_SIZE so a single warp covers the whole head.
  assert(d <= warp_size * VEC_SIZE);
  const uint32_t vec_idx = tx * VEC_SIZE;
  const bool valid = vec_idx < d;

  constexpr uint32_t cp_size = VEC_SIZE * sizeof(T);

  // Shared memory layout (no smem_warp_sums / smem_rms_rcp; reductions are warp-local).
  extern __shared__ char smem_raw[];
  T* smem_weight     = reinterpret_cast<T*>(smem_raw);            // [d]
  T* smem_input_next = smem_weight + d;                            // [NUM_HEADS_PER_ITER * d]

  // Per-thread registers
  float x_cur[VEC_SIZE];
  float sum_sq = 0.f;
  float x_next[VEC_SIZE];
  float sum_sq_next = 0.f;
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; j++) {
    x_cur[j] = 0.f;
    x_next[j] = 0.f;
  }

  // -------------------------------------------------------------------------
  // PDL: wait for grid dependencies
  // -------------------------------------------------------------------------
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // =========================================================================
  // Prologue
  // =========================================================================

  // --- Prologue Step 1: Helper prefetches weight → smem_weight (single head's worth of data) ---
  if constexpr (HAS_WEIGHT) {
    if (is_helper) {
      const uint32_t copy_rounds_w = ceil_div(d, VEC_SIZE * num_helper_threads);
      for (uint32_t i = 0; i < copy_rounds_w; i++) {
        const uint32_t e = (i * num_helper_threads + helper_thread_id) * VEC_SIZE;
        if (e < d) {
          if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
            uint32_t dst_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(smem_weight + e));
            const T* src_ptr = weight + e;
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                         ::"r"(dst_addr), "l"(src_ptr), "n"(cp_size));
          } else {
            vec_t<T, VEC_SIZE> tmp;
            tmp.fill(0.f);
            tmp.load(weight + e);
            tmp.store(smem_weight + e);
          }
        }
      }
    }
  }

  const uint32_t prologue_batch_base = blockIdx.x * NUM_HEADS_PER_ITER;

  // --- Prologue Step 2: Main warp ty loads its head from the first batch ---
  if (!is_helper) {
    vec_t<T, VEC_SIZE> inp_vec;
    inp_vec.fill(0.f);
    const uint32_t my_row = prologue_batch_base + ty;
    if (valid && my_row < num_jobs) {
      const uint32_t batch_idx = my_row / num_heads;
      const uint32_t head_idx  = my_row % num_heads;
      inp_vec.load(input + batch_idx * stride_input_n + head_idx * stride_input_h + vec_idx);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float xi = float(inp_vec[j]);
      x_cur[j]  = xi;
      sum_sq   += xi * xi;
    }
  }

  // --- Prologue Step 3: Helper prefetches next batch (NUM_HEADS_PER_ITER heads) → smem_input_next ---
  if (is_helper) {
    const uint32_t next_batch_base = prologue_batch_base + grid_stride;
    const uint32_t elems_per_head = d / VEC_SIZE;
    const uint32_t total_vecs = NUM_HEADS_PER_ITER * elems_per_head;
    const uint32_t copy_rounds = ceil_div(total_vecs, num_helper_threads);
    for (uint32_t i = 0; i < copy_rounds; i++) {
      const uint32_t flat_vec_idx = i * num_helper_threads + helper_thread_id;
      if (flat_vec_idx < total_vecs) {
        const uint32_t h = flat_vec_idx / elems_per_head;
        const uint32_t e = (flat_vec_idx % elems_per_head) * VEC_SIZE;
        const uint32_t row = next_batch_base + h;
        if (row < num_jobs) {
          const uint32_t batch_idx = row / num_heads;
          const uint32_t head_idx  = row % num_heads;
          if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
            uint32_t dst_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(smem_input_next + h * d + e));
            const T* src_ptr = input + batch_idx * stride_input_n +
                               head_idx * stride_input_h + e;
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                         ::"r"(dst_addr), "l"(src_ptr), "n"(cp_size));
          } else {
            vec_t<T, VEC_SIZE> tmp;
            tmp.fill(0.f);
            tmp.load(input + batch_idx * stride_input_n +
                     head_idx * stride_input_h + e);
            tmp.store(smem_input_next + h * d + e);
          }
        }
      }
    }
    if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
      asm volatile("cp.async.commit_group;\n");
      asm volatile("cp.async.wait_group 0;\n");
    }
  }
  __syncthreads();  // Prologue complete: weight + first prefetched batch visible to all threads

  // =========================================================================
  // Main loop: persistent, step = grid_stride = gridDim.x * NUM_HEADS_PER_ITER
  // =========================================================================
  for (uint32_t batch_base = prologue_batch_base; batch_base < num_jobs;
       batch_base += grid_stride) {
    const bool has_next = (batch_base + grid_stride) < num_jobs;
    const bool has_next_next = (batch_base + 2 * grid_stride) < num_jobs;

    // -----------------------------------------------------------------
    // Sync A: ensure smem_input_next is ready (for DEFER_WAIT, helper
    //         waits on its previously committed cp.async group here).
    //         Prologue already committed+waited, so first iteration's
    //         deferred wait is a no-op.
    // -----------------------------------------------------------------
    if (is_helper) {
      if constexpr (DEFER_WAIT) {
        if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
          asm volatile("cp.async.wait_group 0;\n");
        }
      }
    }
    __syncthreads();  // Sync A

    // -----------------------------------------------------------------
    // Step 1 (main warps): load next batch from smem; warp-reduce sum_sq;
    //                      normalize x_cur and store output; compute x_next.
    // -----------------------------------------------------------------
    vec_t<T, VEC_SIZE> inp_next_vec;
    inp_next_vec.fill(0.f);

    if (!is_helper) {
      // (a) Load this warp's next-batch head from smem
      const uint32_t my_next_row = batch_base + grid_stride + ty;
      if (has_next && valid && my_next_row < num_jobs) {
        inp_next_vec.load(smem_input_next + ty * d + vec_idx);
      }

      // (b) Warp-reduce sum_sq → in-register rms_rcp (one warp = one head)
#pragma unroll
      for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
        sum_sq += math::shfl_xor_sync(sum_sq, offset);
      }
      float rms_rcp = math::rsqrt(sum_sq / float(d) + eps);

      // (c) Normalize x_cur and write output
      const uint32_t my_row = batch_base + ty;
      if (valid && my_row < num_jobs) {
        const uint32_t batch_idx = my_row / num_heads;
        const uint32_t head_idx  = my_row % num_heads;
        vec_t<T, VEC_SIZE> weight_vec, out_vec;
        weight_vec.fill(0.f);
        out_vec.fill(0.f);
        if constexpr (HAS_WEIGHT) {
          weight_vec.load(smem_weight + vec_idx);
#pragma unroll
          for (uint32_t j = 0; j < VEC_SIZE; j++) {
            out_vec[j] = x_cur[j] * rms_rcp * (weight_bias + float(weight_vec[j]));
          }
        } else {
#pragma unroll
          for (uint32_t j = 0; j < VEC_SIZE; j++) {
            out_vec[j] = x_cur[j] * rms_rcp;
          }
        }
        out_vec.store(output + batch_idx * stride_output_n + head_idx * stride_output_h +
                      vec_idx);
      }

      // (d) Compute x_next, sum_sq_next
      sum_sq_next = 0.f;
      if (has_next && my_next_row < num_jobs) {
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
          float xi = float(inp_next_vec[j]);
          x_next[j]    = xi;
          sum_sq_next += xi * xi;
        }
      }

      // (e) Swap current ← next
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        x_cur[j] = x_next[j];
      }
      sum_sq = sum_sq_next;
    }

    __syncthreads();  // Sync B: main done reading smem_input_next; helper may overwrite

    // -----------------------------------------------------------------
    // Step 2 (helper warps): prefetch the next-next batch into smem_input_next.
    //                        No __syncthreads() after — deferred to next iteration's Sync A.
    // -----------------------------------------------------------------
    if (is_helper && has_next_next) {
      const uint32_t nn_batch_base = batch_base + 2 * grid_stride;
      const uint32_t elems_per_head = d / VEC_SIZE;
      const uint32_t total_vecs = NUM_HEADS_PER_ITER * elems_per_head;
      const uint32_t copy_rounds = ceil_div(total_vecs, num_helper_threads);
      for (uint32_t i = 0; i < copy_rounds; i++) {
        const uint32_t flat_vec_idx = i * num_helper_threads + helper_thread_id;
        if (flat_vec_idx < total_vecs) {
          const uint32_t h = flat_vec_idx / elems_per_head;
          const uint32_t e = (flat_vec_idx % elems_per_head) * VEC_SIZE;
          const uint32_t row = nn_batch_base + h;
          if (row < num_jobs) {
            const uint32_t batch_idx = row / num_heads;
            const uint32_t head_idx  = row % num_heads;
            if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
              uint32_t dst_addr = static_cast<uint32_t>(
                  __cvta_generic_to_shared(smem_input_next + h * d + e));
              const T* src_ptr = input + batch_idx * stride_input_n +
                                 head_idx * stride_input_h + e;
              asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                           ::"r"(dst_addr), "l"(src_ptr), "n"(cp_size));
            } else {
              vec_t<T, VEC_SIZE> tmp;
              tmp.fill(0.f);
              tmp.load(input + batch_idx * stride_input_n +
                       head_idx * stride_input_h + e);
              tmp.store(smem_input_next + h * d + e);
            }
          }
        }
      }
      if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
        asm volatile("cp.async.commit_group;\n");
        if constexpr (!DEFER_WAIT) {
          asm volatile("cp.async.wait_group 0;\n");
        }
      }
    }
    // NO sync here — deferred to next iteration's Sync A.
  }  // end persistent loop

  // =========================================================================
  // PDL: signal dependents
  // =========================================================================
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}  // end QKRMSNormPersistentKernel

template <typename T>
cudaError_t QKRMSNorm(T* input, T* weight, T* output, uint32_t batch_size, uint32_t num_heads,
                      uint32_t d, uint32_t stride_input_n, uint32_t stride_input_h,
                      uint32_t stride_output_n, uint32_t stride_output_h, float eps = 1e-5,
                      bool enable_pdl = false, cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);
  float weight_bias = 0.f;

  cudaLaunchConfig_t config;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    const uint32_t num_jobs = batch_size * num_heads;

    // --- Persistent kernel path ---
    // Condition: each warp can process a head in 1 round (d <= 32 * VEC_SIZE)
    const uint32_t rounds_per_warp = ceil_div(d, (uint32_t)VEC_SIZE * 32u);
    bool used_persistent = false;

    if (rounds_per_warp <= 1) {
      constexpr uint32_t NUM_HEADS_PER_ITER = 16;
      constexpr uint32_t NUM_HELPER_WARPS = 4;
      const uint32_t num_warps_total = NUM_HEADS_PER_ITER + NUM_HELPER_WARPS;

      // smem: weight[d] + input_next[NUM_HEADS_PER_ITER * d]
      const uint32_t persistent_smem_size =
          (1u + NUM_HEADS_PER_ITER) * d * static_cast<uint32_t>(sizeof(T));

      // Use a representative kernel for occupancy query
      auto persistent_kernel_fn =
          QKRMSNormPersistentKernel<VEC_SIZE, T, NUM_HEADS_PER_ITER, NUM_HELPER_WARPS, false, true>;

      int num_blocks_per_sm = 0, num_sms = 0, dev_id = 0;
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(persistent_kernel_fn,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                persistent_smem_size));
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &num_blocks_per_sm, persistent_kernel_fn, num_warps_total * 32, persistent_smem_size));
      FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
      FLASHINFER_CUDA_CALL(
          cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

      const uint32_t total_persistent_blocks =
          static_cast<uint32_t>(num_blocks_per_sm * num_sms);

      if (total_persistent_blocks > 0) {
        // Grid size: enough blocks to cover all jobs, capped by hardware
        const uint32_t needed_blocks = ceil_div(num_jobs, NUM_HEADS_PER_ITER);
        const uint32_t num_persistent_blocks =
            std::min(needed_blocks, total_persistent_blocks);
        dim3 nblks_persistent(num_persistent_blocks);
        dim3 nthrs_persistent(32, num_warps_total);
        config.gridDim = nblks_persistent;
        config.blockDim = nthrs_persistent;
        config.dynamicSmemBytes = persistent_smem_size;

        if (num_jobs > 2 * num_persistent_blocks * NUM_HEADS_PER_ITER) {
          // Large workload: defer wait for better overlap
          if (weight != nullptr) {
            auto kernel_fn =
                QKRMSNormPersistentKernel<VEC_SIZE, T, NUM_HEADS_PER_ITER, NUM_HELPER_WARPS, true, true>;
            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kernel_fn,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     persistent_smem_size));
            FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel_fn, input, weight,
                                                   output, d, batch_size, num_heads,
                                                   stride_input_n, stride_input_h,
                                                   stride_output_n, stride_output_h,
                                                   weight_bias, eps));
          } else {
            auto kernel_fn =
                QKRMSNormPersistentKernel<VEC_SIZE, T, NUM_HEADS_PER_ITER, NUM_HELPER_WARPS, true, false>;
            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kernel_fn,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     persistent_smem_size));
            FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel_fn, input, weight,
                                                   output, d, batch_size, num_heads,
                                                   stride_input_n, stride_input_h,
                                                   stride_output_n, stride_output_h,
                                                   weight_bias, eps));
          }
        } else {
          // Small workload: commit + wait together
          if (weight != nullptr) {
            FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, persistent_kernel_fn, input, weight,
                                                   output, d, batch_size, num_heads,
                                                   stride_input_n, stride_input_h,
                                                   stride_output_n, stride_output_h,
                                                   weight_bias, eps));
          } else {
            auto kernel_fn =
                QKRMSNormPersistentKernel<VEC_SIZE, T, NUM_HEADS_PER_ITER, NUM_HELPER_WARPS, false, false>;
            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kernel_fn,
                                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                     persistent_smem_size));
            FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel_fn, input, weight,
                                                   output, d, batch_size, num_heads,
                                                   stride_input_n, stride_input_h,
                                                   stride_output_n, stride_output_h,
                                                   weight_bias, eps));
          }
        }
        used_persistent = true;
      }
    }

    if (!used_persistent) {
      // Fallback: original warp-per-row kernel
      const uint32_t fallback_num_warps = 4;
      const uint32_t smem_size = 0;
      config.dynamicSmemBytes = smem_size;

      if (weight != nullptr) {
        auto kernel = QKRMSNormKernel<VEC_SIZE, T, true>;
        int num_blocks_per_sm = 0, num_sms = 0, dev_id = 0;
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, kernel, fallback_num_warps * 32, smem_size));
        FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
        FLASHINFER_CUDA_CALL(
            cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
        const int needed_blocks = ceil_div(batch_size * num_heads, fallback_num_warps);
        dim3 nblks(std::min(num_blocks_per_sm * num_sms, needed_blocks));
        dim3 nthrs(32, fallback_num_warps);
        config.gridDim = nblks;
        config.blockDim = nthrs;
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, input, weight, output, d,
                                                batch_size, num_heads, stride_input_n,
                                                stride_input_h, stride_output_n,
                                                stride_output_h, weight_bias, eps));
      } else {
        auto kernel = QKRMSNormKernel<VEC_SIZE, T, false>;
        int num_blocks_per_sm = 0, num_sms = 0, dev_id = 0;
        FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, kernel, fallback_num_warps * 32, smem_size));
        FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
        FLASHINFER_CUDA_CALL(
            cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
        const int needed_blocks = ceil_div(batch_size * num_heads, fallback_num_warps);
        dim3 nblks(std::min(num_blocks_per_sm * num_sms, needed_blocks));
        dim3 nthrs(32, fallback_num_warps);
        config.gridDim = nblks;
        config.blockDim = nthrs;
        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, input, weight, output, d,
                                                batch_size, num_heads, stride_input_n,
                                                stride_input_h, stride_output_n,
                                                stride_output_h, weight_bias, eps));
      }
    }
  });

  return cudaSuccess;
}

template <uint32_t VEC_SIZE, typename T>
__global__ void FusedAddRMSNormKernel(T* __restrict__ input, T* __restrict__ residual,
                                      T* __restrict__ weight, const uint32_t d,
                                      const uint32_t stride_input, const uint32_t stride_residual,
                                      float weight_bias, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  extern __shared__ float smem[];
  float* smem_x = smem + ceil_div(num_warps, 4) * 4;
  
  float sum_sq = 0.f;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = 0; i < rounds; i++) {
        vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
        vec_t<T, VEC_SIZE> residual_vec;
    residual_vec.fill(0.f);
    vec_t<float, VEC_SIZE> x_vec;
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = float(input_vec[j]);
      x += float(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = (T)x;
      x_vec[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                         thread_id * VEC_SIZE);
      x_vec.store(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += math::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    vec_t<T, VEC_SIZE> weight_vec;
    vec_t<float, VEC_SIZE> x_vec;
    input_vec.fill(0.f);
    weight_vec.fill(0.f);
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      x_vec.load(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      input_vec[j] = x_vec[j] * rms_rcp * (weight_bias + float(weight_vec[j]));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.store(input + bx * stride_input + i * num_threads * VEC_SIZE +
                      thread_id * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Async-copy one VEC_SIZE-element chunk from global to shared memory:
// cp.async when the chunk is a 4/8/16-byte transaction, synchronous vectorized
// copy otherwise. Callers batch calls into a cp.async group via
// cp.async.commit_group / cp.async.wait_group.
template <uint32_t VEC_SIZE, typename T>
__device__ __forceinline__ void AsyncCopyVecChunk(T* smem_dst, const T* gmem_src) {
  constexpr uint32_t cp_size = VEC_SIZE * sizeof(T);
  if constexpr (cp_size == 16 || cp_size == 8 || cp_size == 4) {
    const uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                 ::"r"(dst_addr), "l"(gmem_src), "n"(cp_size));
  } else {
    vec_t<T, VEC_SIZE> tmp;
    tmp.fill(0.f);
    tmp.load(gmem_src);
    tmp.store(smem_dst);
  }
}

// FusedAddRMSNormPersistentKernel: persistent double-buffered RMS normalization.
//
// Designed for decode workloads (small batch_size, large d) where we want each
// block to process multiple rows in a persistent grid fashion.
//
// One template covers both execution modes, selected by NUM_HELPER_WARPS:
//
// NUM_HELPER_WARPS > 0 — warp-specialized (WS):
//   Main group  (ty < num_warps_main): compute x=inp+res, reduce, normalize
//   Helper group (ty >= num_warps_main): cp.async prefetch + cross-warp reduce
//
// NUM_HELPER_WARPS == 0 — non-warp-specialized (NoWS), for d too large to fit
//   helper warps within the 32-warp block budget (e.g. d = 8192, 16-bit dtypes):
//   every warp both computes and prefetches. Since rounds == 1, each thread
//   owns exactly one VEC_SIZE chunk and cp.async-prefetches the chunk it will
//   later consume itself, so cp.async.wait_group alone provides the required
//   visibility (no cross-thread hand-off, hence no helper warps needed).
//
// Shared memory layout (identical in both modes):
//   [float region, 16-byte aligned]
//     smem_warp_sums[num_warps_main] : per-warp sum_sq for current row
//     smem_rms_rcp[1]                : rms_rcp for current row
//   [T region after alignment]
//     smem_weight[d]                : weight row (prefetched once in prologue)
//     smem_input_next[d]            : next row input  (double buffer)
//     smem_residual_next[d]         : next row residual (double buffer)
//
// Execution flow:
//   Prologue: prefetch weight + row 1 (WS: helper group; NoWS: every thread's
//             own chunk) overlapped with processing row 0 from global
//   Main loop: for each row, pipeline across 3 stages
//     Stage 0: (NoWS: await in-flight prefetch) compute warps reduce current
//              row and load next row data from smem to regs
//     Stage 1: cross-reduce → rms_rcp (WS: helper warp 0; NoWS: warp 0);
//              compute x_next; prefetch row+2*gridDim.x into the double buffer
//              (WS: helper warps 1.., commit+wait; NoWS: own chunk, wait
//              deferred to the next iteration's Stage 0 for maximum overlap)
//     Stage 2: normalize current row from smem_weight; swap cur←next
template <uint32_t VEC_SIZE, typename T, uint32_t NUM_HELPER_WARPS = 4>
__global__ void FusedAddRMSNormPersistentKernel(
    T* __restrict__ input, T* __restrict__ residual, T* __restrict__ weight,
    const uint32_t batch_size, const uint32_t d,
    const uint32_t stride_input, const uint32_t stride_residual,
    float weight_bias, float eps) {
  // NUM_HELPER_WARPS == 0 selects the non-warp-specialized (NoWS) mode
  constexpr bool USE_WS = (NUM_HELPER_WARPS > 0);
  static_assert(NUM_HELPER_WARPS == 0 || NUM_HELPER_WARPS >= 2,
                "WS mode needs >= 2 helper warps (cross-reduce + prefetch)");
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps_total = blockDim.y;
  const uint32_t num_warps_main = num_warps_total - NUM_HELPER_WARPS;
  const bool is_helper = USE_WS && (ty >= num_warps_main);
  const uint32_t helper_ty = ty - num_warps_main;  // meaningful only when is_helper
  const uint32_t num_threads_main = num_warps_main * warp_size;
  const uint32_t main_thread_id = tx + ty * warp_size;
  const uint32_t helper_thread_id = tx + helper_ty * warp_size;
  const uint32_t num_helper_threads = NUM_HELPER_WARPS * warp_size;

  // rounds must be 1: each thread covers exactly one VEC_SIZE chunk
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads_main);
  constexpr uint32_t MAX_ROUNDS = 1;
  assert(rounds <= MAX_ROUNDS);

  // Shared memory layout
  extern __shared__ char smem_raw[];
  const uint32_t float_region_bytes = (num_warps_main + 1) * sizeof(float);
  const uint32_t aligned_float_region = (float_region_bytes + 15u) & ~15u;
  float* smem_warp_sums = reinterpret_cast<float*>(smem_raw);
  float* smem_rms_rcp   = smem_warp_sums + num_warps_main;
  T* smem_weight        = reinterpret_cast<T*>(smem_raw + aligned_float_region);
  T* smem_input_next    = smem_weight + d;
  T* smem_residual_next = smem_input_next + d;

  constexpr uint32_t cp_size = VEC_SIZE * sizeof(T);
  constexpr bool use_cp_async = (cp_size == 16 || cp_size == 8 || cp_size == 4);

  // Per-thread registers for current row (compute threads only)
  float x_cur[VEC_SIZE];
  float sum_sq = 0.f;
  // Per-thread registers for next row (loaded from smem in Stage 0)
  float x_next[VEC_SIZE];
  float sum_sq_next = 0.f;

  // vec index for this thread (used throughout; 1 round only). WS helper
  // threads have main_thread_id >= num_threads_main → vec_idx >= d → invalid.
  const uint32_t vec_idx = main_thread_id * VEC_SIZE;
  const bool valid = vec_idx < d;

  // Copier group for weight / double-buffer prefetch: the helper group in WS
  // mode, every thread in NoWS mode (each thread then copies exactly the
  // chunk it will later consume itself).
  const bool is_copier = USE_WS ? is_helper : true;
  const uint32_t copier_thread_id = USE_WS ? helper_thread_id : main_thread_id;
  const uint32_t num_copier_threads = USE_WS ? num_helper_threads : num_threads_main;

  // -------------------------------------------------------------------------
  // PDL: wait for grid dependencies
  // -------------------------------------------------------------------------
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  // =========================================================================
  // Prologue (executed once before the main loop)
  // =========================================================================

  // --- Prologue Step 1: copiers issue async copies of weight + row 1
  //     (blockIdx.x + gridDim.x) → smem ---
  if (is_copier) {
    const uint32_t next_row = blockIdx.x + gridDim.x;
    const uint32_t copy_rounds = ceil_div(d, VEC_SIZE * num_copier_threads);
    for (uint32_t i = 0; i < copy_rounds; i++) {
      const uint32_t e = (i * num_copier_threads + copier_thread_id) * VEC_SIZE;
      if (e < d) {
        AsyncCopyVecChunk<VEC_SIZE>(smem_weight + e, weight + e);
        if (next_row < batch_size) {
          AsyncCopyVecChunk<VEC_SIZE>(smem_input_next + e,
                                      input + next_row * stride_input + e);
          AsyncCopyVecChunk<VEC_SIZE>(smem_residual_next + e,
                                      residual + next_row * stride_residual + e);
        }
      }
    }
    if constexpr (use_cp_async) {
      asm volatile("cp.async.commit_group;\n");
    }
  }

  // --- Prologue Step 2: compute threads process row 0 from global while the
  //     copies are in flight (WS: different warps run concurrently anyway;
  //     NoWS: each thread overlaps its own cp.async with this compute) ---
  if (!is_helper) {
    vec_t<T, VEC_SIZE> inp_vec, res_vec;
    inp_vec.fill(0.f);
    res_vec.fill(0.f);
    if (valid && blockIdx.x < batch_size) {
      inp_vec.load(input    + blockIdx.x * stride_input    + vec_idx);
      res_vec.load(residual + blockIdx.x * stride_residual + vec_idx);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float xi = float(inp_vec[j]) + float(res_vec[j]);
      x_cur[j]  = xi;
      sum_sq   += xi * xi;
      res_vec[j] = (T)xi;
    }
    if (valid && blockIdx.x < batch_size) {
      res_vec.store(residual + blockIdx.x * stride_residual + vec_idx);
    }
  }

  // --- Prologue Step 3: make weight + row 1 resident; the issuing threads
  //     must wait before the barrier publishes the data to the whole block ---
  if (is_copier) {
    if constexpr (use_cp_async) {
      asm volatile("cp.async.wait_group 0;\n");
    }
  }
  __syncthreads();  // Prologue complete: weight + row1 data visible to all threads

  // =========================================================================
  // Main loop: persistent, step = gridDim.x
  // =========================================================================
  for (uint32_t row = blockIdx.x; row < batch_size; row += gridDim.x) {
    const bool has_next = (row + gridDim.x) < batch_size;
    const bool has_next_next = (row + 2 * gridDim.x) < batch_size;

    // -----------------------------------------------------------------
    // Stage 0:
    //   NoWS: await the prefetch issued in the previous Stage 1 (each thread
    //         only consumes the chunk it issued itself, so wait_group suffices;
    //         WS helpers already waited inside Stage 1 before the barrier)
    //   Compute threads: warp-reduce sum_sq → smem_warp_sums[ty];
    //         if has_next: load next row from smem → regs (inp_next_vec, res_next_vec)
    //   WS helpers: idle
    // -----------------------------------------------------------------
    vec_t<T, VEC_SIZE> inp_next_vec, res_next_vec;
    inp_next_vec.fill(0.f);
    res_next_vec.fill(0.f);

    if constexpr (!USE_WS && use_cp_async) {
      asm volatile("cp.async.wait_group 0;\n");
    }
    if (!is_helper) {
      // Warp-level butterfly reduce
#pragma unroll
      for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
        sum_sq += math::shfl_xor_sync(sum_sq, offset);
      }
      if (tx == 0) {
        smem_warp_sums[ty] = sum_sq;
      }
      // Load next row from smem double buffer to registers
      if (has_next && valid) {
        inp_next_vec.load(smem_input_next    + vec_idx);
        res_next_vec.load(smem_residual_next + vec_idx);
      }
    }
    __syncthreads();  // Stage 0 → Stage 1

    // -----------------------------------------------------------------
    // Stage 1:
    //   Cross-reduce smem_warp_sums → smem_rms_rcp[0]:
    //     WS: helper warp 0 (dedicated); NoWS: warp 0 (also computes below)
    //   Compute threads: if has_next, compute x_next, sum_sq_next,
    //     writeback residual_next
    //   Prefetch row + 2*gridDim.x → smem double buffer:
    //     WS: helper warps 1.., then commit+wait (barrier publishes to main)
    //     NoWS: each thread its own chunk, commit only — the wait is deferred
    //       to the next iteration's Stage 0, overlapping with Stage 2
    // -----------------------------------------------------------------
    sum_sq_next = 0.f;
    const bool is_reducer_warp = USE_WS ? (is_helper && helper_ty == 0) : (ty == 0);
    if (is_reducer_warp) {
      float val = (tx < num_warps_main) ? smem_warp_sums[tx] : 0.f;
#pragma unroll
      for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
        val += math::shfl_xor_sync(val, offset);
      }
      if (tx == 0) {
        smem_rms_rcp[0] = math::rsqrt(val / float(d) + eps);
      }
    }
    if (!is_helper && has_next) {
      // Compute x_next, accumulate sum_sq_next, writeback residual_next
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        float xi = float(inp_next_vec[j]) + float(res_next_vec[j]);
        x_next[j]    = xi;
        sum_sq_next += xi * xi;
        res_next_vec[j] = (T)xi;
      }
      if (valid) {
        res_next_vec.store(residual + (row + gridDim.x) * stride_residual + vec_idx);
      }
    }
    if constexpr (USE_WS) {
      if (is_helper && helper_ty >= 1) {
        // Helper warps 1..: prefetch row + 2*gridDim.x
        if (has_next_next) {
          const uint32_t next_next_row = row + 2 * gridDim.x;
          const uint32_t num_prefetch_threads = (NUM_HELPER_WARPS - 1) * warp_size;
          const uint32_t prefetch_thread_id   = tx + (helper_ty - 1) * warp_size;
          const uint32_t copy_rounds_p = ceil_div(d, VEC_SIZE * num_prefetch_threads);
          for (uint32_t i = 0; i < copy_rounds_p; i++) {
            const uint32_t e = (i * num_prefetch_threads + prefetch_thread_id) * VEC_SIZE;
            if (e < d) {
              AsyncCopyVecChunk<VEC_SIZE>(smem_input_next + e,
                                          input + next_next_row * stride_input + e);
              AsyncCopyVecChunk<VEC_SIZE>(smem_residual_next + e,
                                          residual + next_next_row * stride_residual + e);
            }
          }
        }
        if constexpr (use_cp_async) {
          asm volatile("cp.async.commit_group;\n");
          asm volatile("cp.async.wait_group 0;\n");
        }
      }
    } else {
      if (has_next_next && valid) {
        const uint32_t next_next_row = row + 2 * gridDim.x;
        AsyncCopyVecChunk<VEC_SIZE>(smem_input_next + vec_idx,
                                    input + next_next_row * stride_input + vec_idx);
        AsyncCopyVecChunk<VEC_SIZE>(smem_residual_next + vec_idx,
                                    residual + next_next_row * stride_residual + vec_idx);
      }
      if constexpr (use_cp_async) {
        asm volatile("cp.async.commit_group;\n");
      }
    }
    __syncthreads();  // Stage 1 → Stage 2

    // -----------------------------------------------------------------
    // Stage 2:
    //   Main: read smem_rms_rcp[0], load weight from smem_weight,
    //         normalize x_cur, write to input[row]
    //   Swap: x_cur ← x_next, sum_sq ← sum_sq_next
    // -----------------------------------------------------------------
    if (!is_helper) {
      const float rms_rcp = smem_rms_rcp[0];
      if (valid && row < batch_size) {
        vec_t<T, VEC_SIZE> weight_vec, out_vec;
        weight_vec.fill(0.f);
        out_vec.fill(0.f);
        weight_vec.load(smem_weight + vec_idx);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
          out_vec[j] = x_cur[j] * rms_rcp * (weight_bias + float(weight_vec[j]));
        }
        out_vec.store(input + row * stride_input + vec_idx);
      }
      // Swap current ← next (for last iteration, this is a no-op logically)
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; j++) {
        x_cur[j] = x_next[j];
      }
      sum_sq = sum_sq_next;
    }
    // No __syncthreads() needed here: next iteration's Stage 0 smem write
    // (smem_warp_sums) happens before Stage 1 reads it, protected by the
    // __syncthreads() at the end of Stage 0.
  }  // end persistent loop

  // =========================================================================
  // PDL: signal dependents
  // =========================================================================
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}  // end FusedAddRMSNormPersistentKernel

// Shared launcher for FusedAddRMSNorm / GemmaFusedAddRMSNorm: the two APIs run
// identical kernels and dispatch logic (persistent WS/NoWS + fallback) and
// differ only in weight_bias (0.f: out = x*w; 1.f: out = x*(1+w)).
template <typename T>
cudaError_t FusedAddRMSNormLauncher(T* input, T* residual, T* weight, uint32_t batch_size,
                                    uint32_t d, uint32_t stride_input, uint32_t stride_residual,
                                    float weight_bias, float eps, bool enable_pdl,
                                    cudaStream_t stream) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32);
  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = (ceil_div(num_warps, 4) * 4 + d) * sizeof(float);

  cudaLaunchConfig_t config;
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {

    constexpr uint32_t NUM_HELPER_WARPS = 4;
    const uint32_t num_threads_main = num_warps * 32;
    const uint32_t rounds = ceil_div(d, (uint32_t)VEC_SIZE * num_threads_main);

    // --- Persistent kernel path (highest priority) ---
    // rounds <= 1 is the only hard requirement for the persistent scheme.
    // Warp specialization additionally needs room for NUM_HELPER_WARPS helper
    // warps (d <= 7168 for 16-bit dtypes); otherwise (e.g. d = 8192) fall back
    // to the non-warp-specialized persistent kernel, keeping persistent +
    // double-buffer benefits.
    bool used_persistent = false;
    if (rounds <= 1) {
      const bool use_warp_specialization = (num_warps + NUM_HELPER_WARPS <= 32);
      const uint32_t num_warps_total =
          use_warp_specialization ? num_warps + NUM_HELPER_WARPS : num_warps;
      // Compute persistent kernel smem size:
      //   float region: (num_warps + 1) floats for warp_sums + rms_rcp, 16-byte aligned
      //   T region: 3 * d elements (weight + input_next + residual_next)
      const uint32_t float_region_bytes =
          (num_warps + 1u) * static_cast<uint32_t>(sizeof(float));
      const uint32_t aligned_float_region = (float_region_bytes + 15u) & ~15u;
      const uint32_t persistent_smem_size =
          aligned_float_region + 3u * d * static_cast<uint32_t>(sizeof(T));

      // Same kernel template; NUM_HELPER_WARPS = 0 instantiates the NoWS mode.
      auto persistent_kernel_fn =
          use_warp_specialization
              ? FusedAddRMSNormPersistentKernel<VEC_SIZE, T, NUM_HELPER_WARPS>
              : FusedAddRMSNormPersistentKernel<VEC_SIZE, T, 0>;

      // The probe below (func attribute + occupancy) costs several runtime API
      // calls — more on platforms (e.g. PPU) where those APIs are unsupported and
      // the estimate fallback kicks in. For this T/VEC_SIZE instantiation its
      // result depends only on (device, d), so cache it; otherwise small-batch
      // calls that end up on the fallback kernel pay the probe cost every time.
      static thread_local int cached_dev_id = -1;
      static thread_local uint32_t cached_d = 0;
      static thread_local uint32_t cached_total_blocks = 0;

      int dev_id = 0;
      FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
      uint32_t total_persistent_blocks = 0;
      if (dev_id == cached_dev_id && d == cached_d) {
        total_persistent_blocks = cached_total_blocks;
      } else {
        int num_blocks_per_sm = 0, num_sms = 0;
        // cudaFuncSetAttribute must come before cudaOccupancyMaxActiveBlocksPerMultiprocessor
        // to ensure the smem limit is raised before occupancy is queried. Some platforms
        // (e.g. PPU) do not support this attribute; tolerate the failure as long as the
        // default per-block smem limit already covers the request.
        cudaError_t set_attr_status = cudaFuncSetAttribute(
            persistent_kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
            persistent_smem_size);
        if (set_attr_status != cudaSuccess) {
          int default_smem_per_block = 0;
          FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&default_smem_per_block,
                                                      cudaDevAttrMaxSharedMemoryPerBlock, dev_id));
          if (static_cast<uint32_t>(default_smem_per_block) < persistent_smem_size) {
            return set_attr_status;
          }
          (void)cudaGetLastError();  // clear the sticky error
        }
        FLASHINFER_CUDA_CALL(
            cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
        // Some platforms (e.g. PPU) do not support the occupancy query either; fall
        // back to an estimate from smem/thread limits. Over-estimating residency is
        // safe: the grid-stride loop still covers all rows, extra blocks just queue.
        cudaError_t occ_status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, persistent_kernel_fn, num_warps_total * 32, persistent_smem_size);
        if (occ_status != cudaSuccess) {
          (void)cudaGetLastError();  // clear the sticky error
          int smem_per_sm = 0, max_threads_per_sm = 0;
          FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
              &smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
          FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
              &max_threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, dev_id));
          const int by_smem = smem_per_sm / static_cast<int>(persistent_smem_size);
          const int by_threads = max_threads_per_sm / static_cast<int>(num_warps_total * 32);
          num_blocks_per_sm = std::min(by_smem, by_threads);
        }

        total_persistent_blocks = static_cast<uint32_t>(num_blocks_per_sm * num_sms);
        cached_dev_id = dev_id;
        cached_d = d;
        cached_total_blocks = total_persistent_blocks;
      }

      if (total_persistent_blocks > 0 && batch_size >= total_persistent_blocks) {
        dim3 nblks_persistent(total_persistent_blocks);
        dim3 nthrs_persistent(32, num_warps_total);
        config.gridDim = nblks_persistent;
        config.blockDim = nthrs_persistent;
        config.dynamicSmemBytes = persistent_smem_size;

        FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, persistent_kernel_fn, input, residual,
                                               weight, batch_size, d, stride_input,
                                               stride_residual, weight_bias, eps));
        used_persistent = true;
      }
    }

    if (!used_persistent) {
      // Fallback: original single-row kernel, one block processes one row
      dim3 nblks(batch_size);
      const uint32_t smem_size = (ceil_div(num_warps, 4) * 4 + d) * sizeof(float);
      config.gridDim = nblks;
      config.dynamicSmemBytes = smem_size;

      auto kernel = FusedAddRMSNormKernel<VEC_SIZE, T>;
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, input, residual, weight, d,
                                              stride_input, stride_residual, weight_bias, eps));
    }
  });

  return cudaSuccess;
}

template <typename T>
cudaError_t FusedAddRMSNorm(T* input, T* residual, T* weight, uint32_t batch_size, uint32_t d,
                            uint32_t stride_input, uint32_t stride_residual, float eps = 1e-5,
                            bool enable_pdl = false, cudaStream_t stream = 0) {
  return FusedAddRMSNormLauncher(input, residual, weight, batch_size, d, stride_input,
                                 stride_residual, /*weight_bias=*/0.f, eps, enable_pdl, stream);
}

template <uint32_t VEC_SIZE, typename T, typename O>
__global__ void FusedAddRMSNormQuantKernel(T* __restrict__ input, T* __restrict__ residual,
                                           T* __restrict__ weight, O* __restrict__ output,
                                           const uint32_t d, const uint32_t stride_input,
                                           const uint32_t stride_residual,
                                           const uint32_t stride_output, float weight_bias,
                                           float* scale, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  const uint32_t rounds = ceil_div(d, VEC_SIZE * num_threads);
  const float scale_inv = 1.0f / scale[0];
  extern __shared__ float smem[];
  float* smem_x = smem + ceil_div(num_warps, 4) * 4;

  float sum_sq = 0.f;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, VEC_SIZE> input_vec;
    input_vec.fill(0.f);
    vec_t<T, VEC_SIZE> residual_vec;
    residual_vec.fill(0.f);
    vec_t<float, VEC_SIZE> x_vec;
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      input_vec.load(input + bx * stride_input + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      residual_vec.load(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                        thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      float x = float(input_vec[j]);
      x += float(residual_vec[j]);
      sum_sq += x * x;
      residual_vec[j] = (T)x;
      x_vec[j] = x;
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      residual_vec.store(residual + bx * stride_residual + i * num_threads * VEC_SIZE +
                         thread_id * VEC_SIZE);
      x_vec.store(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += math::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<float, VEC_SIZE> output_vec;
    vec_t<T, VEC_SIZE> weight_vec;
    vec_t<float, VEC_SIZE> x_vec;
    output_vec.fill(0.f);
    weight_vec.fill(0.f);
    x_vec.fill(0.f);
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      weight_vec.load(weight + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
      x_vec.load(smem_x + i * num_threads * VEC_SIZE + thread_id * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; j++) {
      output_vec[j] = x_vec[j] * rms_rcp * (weight_bias + float(weight_vec[j])) * scale_inv;
      output_vec[j] = fmaxf(-448.0f, fminf(output_vec[j], 448.0f));
    }
    if ((i * num_threads + thread_id) * VEC_SIZE < d) {
      output_vec.cast_store(output + bx * stride_output + i * num_threads * VEC_SIZE +
                            thread_id * VEC_SIZE);
    }
  }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, typename O>
cudaError_t FusedAddRMSNormQuant(T* input, T* residual, T* weight, O* output, uint32_t batch_size,
                                 uint32_t d, uint32_t stride_input, uint32_t stride_residual,
                                 uint32_t stride_output, float* scale, float eps = 1e-5,
                                 bool enable_pdl = false, cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32);
  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = (ceil_div(num_warps, 4) * 4 + d) * sizeof(float);
  float weight_bias = 0.f;

  cudaLaunchConfig_t config;
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = FusedAddRMSNormQuantKernel<VEC_SIZE, T, O>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, input, residual, weight, output, d,
                                            stride_input, stride_residual, stride_output,
                                            weight_bias, scale, eps));
  });

  return cudaSuccess;
}

template <typename T>
cudaError_t GemmaRMSNorm(T* input, T* weight, T* output, uint32_t batch_size, uint32_t d,
                         uint32_t stride_input, uint32_t stride_output, float eps = 1e-5,
                         bool enable_pdl = false, cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32);
  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = num_warps * sizeof(float);
  float weight_bias = 1.f;
  void* args[] = {&input, &weight, &output, &d, &stride_input, &stride_output, &weight_bias, &eps};

  cudaLaunchConfig_t config;
  config.gridDim = nblks;
  config.blockDim = nthrs;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    auto kernel = RMSNormKernel<VEC_SIZE, T>;
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, input, weight, output, d, stride_input,
                                            stride_output, weight_bias, eps));
  });
  return cudaSuccess;
}

template <typename T>
cudaError_t GemmaFusedAddRMSNorm(T* input, T* residual, T* weight, uint32_t batch_size, uint32_t d,
                                 uint32_t stride_input, uint32_t stride_residual, float eps = 1e-5,
                                 bool enable_pdl = false, cudaStream_t stream = 0) {
  // Same kernels/dispatch as FusedAddRMSNorm; weight_bias = 1.f gives the
  // Gemma-style out = x * (1 + w) normalization.
  return FusedAddRMSNormLauncher(input, residual, weight, batch_size, d, stride_input,
                                 stride_residual, /*weight_bias=*/1.f, eps, enable_pdl, stream);
}

template <typename T>
struct QuantTypeStaticVals;

template <>
struct QuantTypeStaticVals<int8_t> {
  static constexpr float MAX_VAL = 127.f;
  static constexpr float MIN_SCALING_FACTOR = 0.f;
  static constexpr float MIN_SCALING_FACTOR_RCP = FLT_MAX;
};

template <typename Tf, typename T>
__inline__ __device__ Tf compute_layernorm(Tf val, float s_mean, float s_variance, T const* gemma,
                                           T const* beta, int i) {
  Tf ret = (val - s_mean) * s_variance * cuda_cast<Tf>(gemma[i]);
  if (beta != nullptr) {
    ret = ret + cuda_cast<Tf>(beta[i]);
  }
  return ret;
}

template <typename T, typename Tw, typename QuantT, bool USE_SHMEM,
          bool USE_DIFF_OF_SQUARES = false>
__global__ void generalLayerNorm(T const* input, Tw const* gemma, Tw const* beta, T* normed_output,
                                 float const eps, int tokens, int hidden_dim,
                                 float const* clamp_ptr, float const* scale_orig_quant_per_tensor,
                                 float* scale_orig_quant_per_token, float* sum_per_token,
                                 QuantT* normed_output_quant, bool has_fp8_min_scaling) {
  constexpr auto num_elems_T = num_elems<T>::value;
  using QuantT_packed_t = typename packed_as<QuantT, num_elems_T>::type;
  using float_packed_t = typename packed_as<float, num_elems_T>::type;
  using T_scalar = typename packed_as<T, 1>::type;

  // The clamping minimum / maximum values.
  T const clamp_min = cuda_cast<T>(clamp_ptr ? clamp_ptr[0] : -FLT_MAX);
  T const clamp_max = cuda_cast<T>(clamp_ptr ? clamp_ptr[1] : FLT_MAX);

  // The quantized data type's maximum value (upper-bound).
  static constexpr float MAX_QUANT_VAL = QuantTypeStaticVals<QuantT>::MAX_VAL;
  // The minimum scaling factor (lower-bound)
  static constexpr float MIN_SCALING_FACTOR = QuantTypeStaticVals<QuantT>::MIN_SCALING_FACTOR;
  static constexpr float MIN_SCALING_FACTOR_RCP =
      QuantTypeStaticVals<QuantT>::MIN_SCALING_FACTOR_RCP;

  extern __shared__ __align__(sizeof(float)) char _shmem[];
  T* shmem = reinterpret_cast<T*>(_shmem);
  __shared__ float s_mean;
  __shared__ float s_variance;

  int const tidx = threadIdx.x;
  int const bidx = blockIdx.x;

  float mean = 0.0f;
  float variance = 0.0f;
  float local_sum = 0.0f;
  float local_var_sum = 0.0f;

  int const n_elems = hidden_dim / num_elems_T;
  for (int i = tidx; i < n_elems; i += blockDim.x) {
    const T val = input[bidx * n_elems + i];
    if constexpr (USE_SHMEM) {
      shmem[i] = val;
    }

    const float_packed_t val_f = cuda_cast<float_packed_t>(val);
    local_sum += cuda_sum<float>(val_f);
    if constexpr (USE_DIFF_OF_SQUARES) {
      local_var_sum += cuda_sum<float>(val_f * val_f);
    }
  }

  if constexpr (USE_DIFF_OF_SQUARES) {
    float packed[2] = {local_sum, local_var_sum};
    blockReduceSumV2<float, 2>(packed);
    mean = packed[0];
    variance = packed[1];
  } else {
    mean = blockReduceSum(local_sum);
  }

  if (threadIdx.x == 0) {
    mean = mean / hidden_dim;
    s_mean = mean;
    if constexpr (USE_DIFF_OF_SQUARES) {
      variance = (variance / hidden_dim) - (mean * mean);  // Var[x] = E[x²] - E[x]²
      s_variance = rsqrtf(variance + eps);
    }
  }
  __syncthreads();

  if constexpr (!USE_DIFF_OF_SQUARES) {
    for (int i = tidx; i < n_elems; i += blockDim.x) {
      const T val = USE_SHMEM ? shmem[i] : input[bidx * n_elems + i];
      float_packed_t diff = cuda_cast<float_packed_t>(val) - s_mean;
      local_var_sum += cuda_sum<float>(diff * diff);
    }
    variance = blockReduceSum(local_var_sum);

    if (threadIdx.x == 0) {
      s_variance = rsqrtf(variance / hidden_dim + eps);
    }
    __syncthreads();
  }

  bool const with_per_token_scaling = scale_orig_quant_per_token != nullptr;
  bool const with_per_tensor_scaling = scale_orig_quant_per_tensor != nullptr;
  bool const with_per_token_sum = sum_per_token != nullptr;

  const float_packed_t scale_orig_quant =
      cuda_cast<float_packed_t>(with_per_tensor_scaling ? *scale_orig_quant_per_tensor : 0.0f);
  T_scalar amax = 1e-6f;
  local_sum = 0.f;

  for (int i = tidx; i < n_elems; i += blockDim.x) {
    int const index = bidx * n_elems + i;
    const float_packed_t val_f = cuda_cast<float_packed_t>(USE_SHMEM ? shmem[i] : input[index]);
    T val = cuda_cast<T>(compute_layernorm(val_f, s_mean, s_variance, gemma, beta, i));

    if (with_per_token_scaling) {
      val = cuda_clamp(val, clamp_min, clamp_max);
      amax = cuda_max(cuda_max<T_scalar, T>(cuda_abs(val)), amax);
      if constexpr (USE_SHMEM) {
        shmem[i] = val;
      }
    } else if (with_per_tensor_scaling) {
      val = cuda_clamp(val, clamp_min, clamp_max);
      reinterpret_cast<QuantT_packed_t*>(normed_output_quant)[index] =
          cuda_cast<QuantT_packed_t>(cuda_cast<float_packed_t>(val) * scale_orig_quant);
    } else {
      normed_output[index] = val;
    }

    if (with_per_token_sum) {
      local_sum += cuda_sum<float>(cuda_cast<float_packed_t>(val));
    }
  }

  if (with_per_token_scaling) {
    float abs_max_f = blockAllReduceMax(cuda_cast<float>(amax));
    float const dynamic_per_token_scale =
        has_fp8_min_scaling ? fminf(MAX_QUANT_VAL / abs_max_f, MIN_SCALING_FACTOR_RCP)
                            : (MAX_QUANT_VAL / abs_max_f);
    for (int i = tidx; i < n_elems; i += blockDim.x) {
      int const index = bidx * n_elems + i;
      float_packed_t val_f = cuda_cast<float_packed_t>(USE_SHMEM ? shmem[i] : input[index]);
      if constexpr (!USE_SHMEM) {
        val_f = compute_layernorm(val_f, s_mean, s_variance, gemma, beta, i);
      }

      reinterpret_cast<QuantT_packed_t*>(normed_output_quant)[index] =
          cuda_cast<QuantT_packed_t>(val_f * cuda_cast<float_packed_t>(dynamic_per_token_scale));
    }
    if (tidx == 0) {
      scale_orig_quant_per_token[bidx] =
          has_fp8_min_scaling ? cuda_max(abs_max_f / MAX_QUANT_VAL, MIN_SCALING_FACTOR)
                              : abs_max_f / MAX_QUANT_VAL;
    }
  }

  if (with_per_token_sum) {
    float packed_sum[1] = {local_sum};
    blockReduceSumV2<float, 1>(packed_sum);
    if (tidx == 0) {
      sum_per_token[bidx] = packed_sum[0];
    }
  }
}

template <bool USE_DIFF_OF_SQUARES, typename T, typename Tw, typename QuantT>
void dispatch_layernorm_type_square_method(
    T const* input, Tw const* gemma, Tw const* beta, T* normed_output, float const eps, int tokens,
    int hidden_dim, float const* clamp_ptr, float const* scale_orig_quant_per_tensor,
    float* scale_orig_quant_per_token, float* sum_per_token, QuantT* normed_output_quant,
    bool const has_fp8_min_scaling, dim3 const grid, dim3 const block, size_t const shmem_size,
    cudaStream_t stream) {
  // Do we use shared memory to cache intermediate results
  bool use_shmem = true;
  if (shmem_size >= (48 << 10)) {
    cudaError_t ret =
        cudaFuncSetAttribute(generalLayerNorm<T, Tw, QuantT, true, USE_DIFF_OF_SQUARES>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    // Use shared memory when the capacity is enough
    use_shmem = (ret == cudaSuccess);
  }

  if (use_shmem) {
    generalLayerNorm<T, Tw, QuantT, true, USE_DIFF_OF_SQUARES><<<grid, block, shmem_size, stream>>>(
        input, gemma, beta, normed_output, eps, tokens, hidden_dim, clamp_ptr,
        scale_orig_quant_per_tensor, scale_orig_quant_per_token, sum_per_token, normed_output_quant,
        has_fp8_min_scaling);
  } else {
    generalLayerNorm<T, Tw, QuantT, false, USE_DIFF_OF_SQUARES><<<grid, block, 0, stream>>>(
        input, gemma, beta, normed_output, eps, tokens, hidden_dim, clamp_ptr,
        scale_orig_quant_per_tensor, scale_orig_quant_per_token, sum_per_token, normed_output_quant,
        has_fp8_min_scaling);
  }
}

template <typename T, typename Tw, typename QuantT>
void dispatch_layernorm_type(T const* input, Tw const* gemma, Tw const* beta, T* normed_output,
                             float const eps, int tokens, int hidden_dim, float const* clamp_ptr,
                             float const* scale_orig_quant_per_tensor,
                             float* scale_orig_quant_per_token, float* sum_per_token,
                             QuantT* normed_output_quant, bool const has_fp8_min_scaling,
                             dim3 const grid, dim3 const block, size_t const shmem_size,
                             cudaStream_t stream, bool const use_diff_of_squares) {
  if (use_diff_of_squares) {
    dispatch_layernorm_type_square_method<true>(
        input, gemma, beta, normed_output, eps, tokens, hidden_dim, clamp_ptr,
        scale_orig_quant_per_tensor, scale_orig_quant_per_token, sum_per_token, normed_output_quant,
        has_fp8_min_scaling, grid, block, shmem_size, stream);
  } else {
    dispatch_layernorm_type_square_method<false>(
        input, gemma, beta, normed_output, eps, tokens, hidden_dim, clamp_ptr,
        scale_orig_quant_per_tensor, scale_orig_quant_per_token, sum_per_token, normed_output_quant,
        has_fp8_min_scaling, grid, block, shmem_size, stream);
  }
}

template <typename T, typename Tw>
cudaError_t LayerNorm(T* input, Tw* gemma, Tw* beta, T* out, uint32_t tokens, uint32_t hidden_dim,
                      float eps = 1e-5, cudaStream_t stream = 0) {
  dim3 grid(tokens);
  dim3 block(min(hidden_dim, 1024));
  // Make sure block.x is multiple of 32 for warp shuffle to work
  block.x = 32 * ((block.x + 31) / 32);

  constexpr size_t vec_size = 2;
  const size_t shmem_size = hidden_dim * sizeof(T);
  bool const use_vec_type = (hidden_dim % vec_size == 0) &&
                            (std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value);

  // Enable min_scaling factor if it is fp8 row-wise per-token quantization
  // TODO(kaixih): add support for fp8 quantization if needed
  bool has_fp8_min_scaling = false;
  float* clamp_ptr = nullptr;
  float* scale = nullptr;
  float* dynamic_scale = nullptr;
  float* sum_per_token = nullptr;
  int8_t* normed_output_quant = nullptr;
  bool use_diff_of_squares = false;

  if (use_vec_type) {
    using Tp = typename packed_as<T, vec_size>::type;
    using Twp = typename packed_as<Tw, vec_size>::type;
    dispatch_layernorm_type(reinterpret_cast<Tp const*>(input), reinterpret_cast<Twp const*>(gemma),
                            reinterpret_cast<Twp const*>(beta), reinterpret_cast<Tp*>(out), eps,
                            tokens, hidden_dim, clamp_ptr, scale, dynamic_scale, sum_per_token,
                            normed_output_quant, has_fp8_min_scaling, grid, block, shmem_size,
                            stream, use_diff_of_squares);
  } else {
    dispatch_layernorm_type(input, gemma, beta, out, eps, tokens, hidden_dim, clamp_ptr, scale,
                            dynamic_scale, sum_per_token, normed_output_quant, has_fp8_min_scaling,
                            grid, block, shmem_size, stream, use_diff_of_squares);
  }
  return cudaSuccess;
}

}  // namespace norm

}  // namespace flashinfer

#endif  // FLASHINFER_NORM_CUH_
