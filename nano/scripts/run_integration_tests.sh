#!/bin/bash

export HYDRA_FULL_ERROR=1
NC='\033[0m'
Red='\033[0;31m'   
Green='\033[0;32m'  


status=0 # Will become 1 if any test fails
test_name="Checkpointing WORLD_SIZE=1"
echo "=== Running test: '${test_name}' ===" 
python -m tests.integration.test_checkpoint_loading_same_world_size > integration_tests_output.log  2>&1
ws1_exit=$? 
if [ $ws1_exit -eq 0 ]; then 
    echo -e "${Green}[PASS]${NC} ${test_name}" 
else 
    echo -e "${Red}[FAIL]${NC} ${test_name}" 
    status=1 
fi

test_name="Checkpointing WORLD_SIZE=2"
echo "=== Running test: '${test_name}' ===" 
torchrun --standalone --nproc-per-node=2 -m tests.integration.test_checkpoint_loading_same_world_size >> integration_tests_output.log  2>&1
ws2_exit=$? 
if [ $ws2_exit -eq 0 ]; then 
    echo -e "${Green}[PASS]${NC} ${test_name}" 
else 
    echo -e "${Red}[FAIL]${NC} ${test_name}" 
    status=1 
fi


test_name="Checkpointing mix world_sizes (descending)"
echo "=== Running test: '${test_name}' ===" 
torchrun --standalone --nproc-per-node=2  -m tests.integration.test_checkpoint_mix_world_sizes_desc >> integration_tests_output.log  2>&1
ws_mix_desc_exit=$? 
if [ $ws_mix_desc_exit -eq 0 ]; then 
    echo -e "${Green}[PASS]${NC} ${test_name}" 
else 
    echo -e "${Red}[FAIL]${NC} ${test_name}" 
    status=1 
fi


test_name="Checkpointing mix world_sizes (ascending)"
echo "=== Running test: '${test_name}' ===" 
torchrun --standalone --nproc-per-node=2  -m tests.integration.test_checkpoint_mix_world_sizes_asc >> integration_tests_output.log  2>&1
ws_mix_desc_exit=$? 
if [ $ws_mix_desc_exit -eq 0 ]; then 
    echo -e "${Green}[PASS]${NC} ${test_name}" 
else 
    echo -e "${Red}[FAIL]${NC} ${test_name}" 
    status=1 
fi


test_name="Test residual metrics with gradient accumulation 1"
echo "=== Running test: '${test_name}' ===" 
torchrun --standalone --nproc-per-node=2  -m tests.test_residual_metrics --test grad_acc_1  >> integration_tests_output.log  2>&1
ws_mix_desc_exit=$? 
if [ $ws_mix_desc_exit -eq 0 ]; then 
    echo -e "${Green}[PASS]${NC} ${test_name}" 
else 
    echo -e "${Red}[FAIL]${NC} ${test_name}" 
    status=1 
fi

test_name="Test residual metrics with gradient accumulation 2"
echo "=== Running test: '${test_name}' ===" 
torchrun --standalone --nproc-per-node=2  -m tests.test_residual_metrics --test grad_acc_2  >> integration_tests_output.log  2>&1
ws_mix_desc_exit=$? 
if [ $ws_mix_desc_exit -eq 0 ]; then 
    echo -e "${Green}[PASS]${NC} ${test_name}" 
else 
    echo -e "${Red}[FAIL]${NC} ${test_name}" 
    status=1 
fi

exit $status