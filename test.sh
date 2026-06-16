#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="./"

test_list=("tests/attention/test_batch_decode_kernels" "tests/attention/test_batch_prefill_kernels" "tests/attention/test_deepseek_mla" "tests/attention/test_mla_decode_kernel" "tests/attention/test_rope" "tests/attention/test_batch_prefill" "tests/attention/test_mla_page" "tests/utils/test_activation" "tests/utils/test_sampling" "tests/utils/test_norm" "tests/attention/test_mla_page")

for test_name in "${test_list[@]}"; do
    log_file="utlogs/${test_name}.log"
    echo "Running: pytest ${test_name}.py -v"
    echo "Log file: ${log_file}"

    pytest "${test_name}.py" -v > "${log_file}" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ ${test_name} passed"
    else
        echo "✗ ${test_name} failed (exit code: $?)"
    fi
    echo ""
done

echo "All tests completed. Logs saved in utlogs/"
