#!/bin/bash

# Profile sweep: 3 models × 2 modes = 6 runs

cd /teamspace/studios/this_studio/Flash-Attention-2-Triton

for model in small medium large; do
    echo "=== Profiling $model forward_only ==="
    uv run /opt/nvidia/nsight-compute/2024.3.2/host/target-linux-x64/nsys profile -o profile_${model}_forward_only --capture-range=cudaProfilerApi \
        uv run python cs336_systems/benchmarking.py --model $model --forward_only

    echo "=== Profiling $model forward_backward ==="
    uv run /opt/nvidia/nsight-compute/2024.3.2/host/target-linux-x64/nsys profile -o profile_${model}_forward_backward --capture-range=cudaProfilerApi \
        uv run python cs336_systems/benchmarking.py --model $model
done

echo "=== All profiling complete ==="
ls -la profile_*.nsys-rep

