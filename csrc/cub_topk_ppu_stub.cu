/*
 * Copyright (c) 2026 by FlashInfer team.
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

// The PPU SDK lacks DeviceBatchedTopK's execution-environment headers.
// Keep the FFI symbols available so radix top-k can build independently.

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace {

[[noreturn]] void raise_cub_topk_unavailable() {
  TVM_FFI_ICHECK(false)
      << "cub_topk is unavailable in the PPU build: the CUB "
         "DeviceBatchedTopK backend requires execution-environment "
         "headers the PPU toolchain does not ship. Use the radix top-k "
         "backend (FLASHINFER_TOPK_ALGO=default) instead.";
  throw;  // unreachable; silences missing-return warnings
}

}  // namespace

void cub_topk_page_table_transform(
    TensorView input, TensorView output_page_table, TensorView src_page_table,
    TensorView lengths, Optional<TensorView> maybe_output_raw_indices,
    Optional<TensorView> maybe_workspace_buffer, int64_t top_k, int64_t tie_break,
    int64_t page_size, Optional<TensorView> maybe_row_to_batch,
    Optional<TensorView> maybe_row_starts, Optional<TensorView> maybe_page_table_row_starts) {
  raise_cub_topk_unavailable();
}

int64_t cub_topk_page_table_transform_workspace_size(TensorView input, TensorView lengths,
                                                     int64_t top_k, int64_t tie_break,
                                                     bool with_raw_indices,
                                                     bool with_row_starts) {
  raise_cub_topk_unavailable();
}

void cub_topk_ragged_transform(TensorView input, TensorView output_indices, TensorView offsets,
                               TensorView lengths, Optional<TensorView> maybe_workspace_buffer,
                               int64_t top_k, int64_t tie_break,
                               Optional<TensorView> maybe_row_starts) {
  raise_cub_topk_unavailable();
}

int64_t cub_topk_ragged_transform_workspace_size(TensorView input, TensorView lengths,
                                                 int64_t top_k, int64_t tie_break,
                                                 bool with_row_starts) {
  raise_cub_topk_unavailable();
}

void cub_topk(TensorView input, TensorView output_indices, TensorView output_values,
              Optional<TensorView> maybe_workspace_buffer, int64_t top_k, int64_t tie_break) {
  raise_cub_topk_unavailable();
}

int64_t cub_topk_workspace_size(TensorView input, int64_t top_k, int64_t tie_break) {
  raise_cub_topk_unavailable();
}
