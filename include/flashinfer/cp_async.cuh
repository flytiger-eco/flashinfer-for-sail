/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef FLASHINFER_CP_ASYNC_CUH_
#define FLASHINFER_CP_ASYNC_CUH_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "macros.cuh"

#include <cstdint>

namespace flashinfer {

namespace cp_async {

enum class SharedMemFillMode {
  kFillZero,  // Fill zero to shared memory when predicate is false
  kNoFill     // Do not fill zero to shared memory when predicate is false
};

enum class PrefetchMode {
  kNoPrefetch,  // Do not fetch additional data from global memory to L2
  kPrefetch     // Fetch additional data from global memory to L2
};

#if SAIL_ARCH_SM80_AIU
struct cube_load_params {
    // tensor_h
    int32_t tensor_dims_y;
    // tensor_w
    int32_t tensor_dims_x;
    // tensor_z
    int32_t tensor_dims_z;
    // y_offset
    int32_t y_offset;
    // x_offset
    int32_t x_offset;
    // z_offset, swizzle: usigned 15 bits; linear: signed 16 bits;
    int32_t z_offset;
    // cube_h
    uint16_t cube_h;
    // cube_w
    uint16_t cube_w;
    // cube_z
    uint16_t cube_z;
    // cube_n
    uint16_t cube_n;

    uint32_t channel_offset;
    uint32_t start_coord_y;
    uint32_t start_coord_x;

    uint32_t slice_num;

    cube_load_params ():slice_num(4), tensor_dims_y(0), tensor_dims_x(0), tensor_dims_z(0), y_offset(0), x_offset(0), z_offset(0), cube_h(0), cube_w(0), cube_z(0), cube_n(1), channel_offset(0),start_coord_y(0),start_coord_x(0)
    {};


    cube_load_params operator=(const cube_load_params &t)
    {
        tensor_dims_y= t.tensor_dims_y;
        tensor_dims_x= t.tensor_dims_x;
        tensor_dims_z= t.tensor_dims_z;
        y_offset= t.y_offset;
        x_offset= t.x_offset;
        z_offset= t.z_offset;
        cube_h= t.cube_h;
        cube_w= t.cube_w;
        cube_z= t.cube_z;
        cube_n= t.cube_n;
        return *this;
    };
};
#endif

#if (__CUDACC_VER_MAJOR__ >= 11)
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800))
#define FLASHINFER_CP_ASYNC_ENABLED
#endif
#endif

/*!
 * \brief Wrapper of PTX cp.async.commit_group instruction, commit all prior uncommitted
 *   cp.async instructions to a group
 */
__device__ __forceinline__ void commit_group() {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.wait_group instruction
 * \tparam n Wait till most recent n groups are committed
 */
template <size_t n>
__device__ __forceinline__ void wait_group() {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.cg.shared.global instruction, asynchronously copy data from
 *   global memory to shared memory
 * \tparam prefetch_mode Whether to fetch additional data from global memory to L2
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 */
template <PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load_128b(T* smem_ptr, const T* gmem_ptr) {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(16), "r"(16));
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(16), "r"(16));
  }
#else
  *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
#endif
}

#if SAIL_ARCH_SM80_AIU
template <typename Dtype, bool trans, typename T>
__device__ __forceinline__ void load_128b_aiu(T* smem_ptr, const T* gmem_ptr, cube_load_params* q_AIU_para) {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
if constexpr (trans){
  if constexpr (sizeof(Dtype)==2) { 
      asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.trans.padz.swzl.2d.b16 [%0], [%1], "
      "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_int_ptr), "l"(gmem_ptr), 
          "r"(q_AIU_para->tensor_dims_y), "r"(q_AIU_para->tensor_dims_x), "r"(q_AIU_para->tensor_dims_z), 
          "r"(q_AIU_para->y_offset), "r"(q_AIU_para->x_offset), "r"(q_AIU_para->z_offset),
          "r"(q_AIU_para->cube_h), "r"(q_AIU_para->cube_w), "r"(q_AIU_para->cube_z), "r"(q_AIU_para->cube_n));
  }else{
    //b8 not support trans
      asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b8 [%0], [%1], "
      "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        ::"r"(smem_int_ptr), "l"(gmem_ptr), 
          "r"(q_AIU_para->tensor_dims_y), "r"(q_AIU_para->tensor_dims_x), "r"(q_AIU_para->tensor_dims_z), 
          "r"(q_AIU_para->y_offset), "r"(q_AIU_para->x_offset), "r"(q_AIU_para->z_offset),
          "r"(q_AIU_para->cube_h), "r"(q_AIU_para->cube_w), "r"(q_AIU_para->cube_z), "r"(q_AIU_para->cube_n));
  }
}else{
  if constexpr (sizeof(Dtype)==2) { 
      asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b16 [%0], [%1], "
      "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        :: "r"(smem_int_ptr), "l"(gmem_ptr), 
          "r"(q_AIU_para->tensor_dims_y), "r"(q_AIU_para->tensor_dims_x), "r"(q_AIU_para->tensor_dims_z), 
          "r"(q_AIU_para->y_offset), "r"(q_AIU_para->x_offset), "r"(q_AIU_para->z_offset),
          "r"(q_AIU_para->cube_h), "r"(q_AIU_para->cube_w), "r"(q_AIU_para->cube_z), "r"(q_AIU_para->cube_n));
  }else{
      asm volatile("ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.2d.b8 [%0], [%1], "
      "{%2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11};\n"
        ::"r"(smem_int_ptr), "l"(gmem_ptr), 
          "r"(q_AIU_para->tensor_dims_y), "r"(q_AIU_para->tensor_dims_x), "r"(q_AIU_para->tensor_dims_z), 
          "r"(q_AIU_para->y_offset), "r"(q_AIU_para->x_offset), "r"(q_AIU_para->z_offset),
          "r"(q_AIU_para->cube_h), "r"(q_AIU_para->cube_w), "r"(q_AIU_para->cube_z), "r"(q_AIU_para->cube_n));
  }
}

#else
  *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
#endif
}
#endif

/*!
 * \brief Wrapper of PTX cp.async.cg.shared.global instruction, asynchronously copy data from
 *   global memory to shared memory with predicate.
 * \tparam prefetch_mode Whether to fetch additional data from global memory to L2
 * \tparam fill_mode Whether to fill zero to shared memory when predicate is false
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 * \param predicate Predicate value
 * \note fill zero is slower than not fill zero
 */
template <PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load_128b(T* smem_ptr, const T* gmem_ptr, bool predicate) {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    int src_in_bytes = predicate ? 16 : 0;
    if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
      asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                   "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
    } else {
      asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                   "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
    }
  } else {
    if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
    } else {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
    }
  }
#else
  if (predicate) {
    *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
  } else {
    if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
      *((uint4*)smem_ptr) = make_uint4(0, 0, 0, 0);
    }
  }
#endif
}

/*!
 * \brief Load specified number of bits per thread from global memory to shared memory
 * \tparam num_bits Number of bits to load, must be 128 or 256
 * \tparam prefetch_mode Whether to fetch additional data from global memory to L2
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 */
template <size_t num_bits, PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load(T* smem_ptr, const T* gmem_ptr) {
  static_assert(num_bits == 128 || num_bits == 256, "num_bits must be 128 or 256");
  if constexpr (num_bits == 128) {
    load_128b<prefetch_mode>(smem_ptr, gmem_ptr);
  } else {
    load_128b<prefetch_mode>(smem_ptr, gmem_ptr);
    load_128b<prefetch_mode>(smem_ptr + 16 / sizeof(T), gmem_ptr + 16 / sizeof(T));
  }
}

/*!
 * \brief Load specified number of bits per thread from global memory to shared memory with
 *   predicate
 * \tparam num_bits Number of bits to load, must be 128 or 256
 * \tparam prefetch_mode Whether to fetch additional data from global memory to L2
 * \tparam fill_mode Whether to fill zero to shared memory when predicate is false
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 * \param predicate Predicate value
 * \note fill zero is slower than not fill zero
 */
template <size_t num_bits, PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load(T* smem_ptr, const T* gmem_ptr, bool predicate) {
  static_assert(num_bits == 128 || num_bits == 256, "num_bits must be 128 or 256");
  if constexpr (num_bits == 128) {
    pred_load_128b<prefetch_mode, fill_mode>(smem_ptr, gmem_ptr, predicate);
  } else {
    pred_load_128b<prefetch_mode, fill_mode>(smem_ptr, gmem_ptr, predicate);
    pred_load_128b<prefetch_mode, fill_mode>(smem_ptr + 16 / sizeof(T), gmem_ptr + 16 / sizeof(T),
                                             predicate);
  }
}

/*!
 * \brief Like pred_load_128b but reads only 64 bits from global memory into the lower 64 bits of
 *   the 128-bit shared memory destination, zero-padding the upper 64 bits when predicate is true.
 *   Used for NVFP4 KV loading: GMEM stores 2 FP4 elements per byte (packed), so each SMEM
 *   128-bit slot is loaded from 64 GMEM bits and padded with 64 bits of zeros.
 */
template <PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load_128b_from_64b(T* smem_ptr, const T* gmem_ptr,
                                                        bool predicate) {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    int src_in_bytes = predicate ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(8), "r"(src_in_bytes));

  } else {
    // kNoFill: only issue the copy if predicate is true; cp.async always zeros the upper 8 bytes
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, %0, 0;\n"
        " @p cp.async.ca.shared.global [%1], [%2], %3, %4;\n"
        "}\n" ::"r"((int)predicate),
        "r"(smem_int_ptr), "l"(gmem_ptr), "n"(8), "n"(8));
  }
#else
  if (predicate) {
    uint64_t* smem_u64 = reinterpret_cast<uint64_t*>(smem_ptr);
    smem_u64[0] = *reinterpret_cast<const uint64_t*>(gmem_ptr);
    smem_u64[1] = 0;
  } else {
    if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
      *((uint4*)smem_ptr) = make_uint4(0, 0, 0, 0);
    }
  }
#endif
}

/*!
 * \brief Asynchronously copy 4 bytes from global memory to shared memory (LDGSTS.32).
 *   Uses cp.async.ca.shared.global with a 4-byte transfer size.
 *   When fill_mode==kFillZero and predicate is false, writes 0 to shared memory.
 *   When fill_mode==kNoFill and predicate is false, no operation is issued.
 * \tparam fill_mode Whether to fill zero to shared memory when predicate is false.
 * \param smem_ptr 4-byte aligned shared memory destination.
 * \param gmem_ptr Global memory source.
 * \param predicate Predicate value.
 */
template <SharedMemFillMode fill_mode>
__device__ __forceinline__ void pred_load_32b(uint32_t* smem_ptr, const uint32_t* gmem_ptr,
                                              bool predicate) {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    int src_in_bytes = predicate ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(4), "r"(src_in_bytes));
  } else {
    asm volatile(
        "{\n"
        " .reg .pred p;\n"
        " setp.ne.b32 p, %0, 0;\n"
        " @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int)predicate),
        "r"(smem_int_ptr), "l"(gmem_ptr), "n"(4));
  }
#else
  if (predicate) {
    *smem_ptr = *gmem_ptr;
  } else if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    *smem_ptr = 0;
  }
#endif
}

}  // namespace cp_async

}  // namespace flashinfer

#endif  // FLASHINFER_CP_ASYNC_CUH_
