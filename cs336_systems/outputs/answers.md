Problem (benchmarking_script)
Answer:
Refer: [Results](cs336_systems/outputs/benchmark_timings.json)
Timing plot: [Timing Plot](cs336_systems/outputs/benchmark_timings.png)
Overall very little std deviation, and model increase leads to more time taken for both forward and backward pass. 2.7B vs xl we observe that xl is slower than 2.7B. This maybe due to lesser num layers in 2.7B model.
If no warmup is done, the passes have both higher std deviation and higher average time as expected.

Problem (nsys_profile):
a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?
Answer: We will focus only on large mdoel with context length 32 for ease of analysis. Experimented on L4 gpu
Around 113.5 ms for forward pass in python standard library and 142.87 ms for forward pass in nsys profile.
The difference is due to the overhead of the nsys profile.

b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes? (Hint: look at the “CUDA GPU Kernel Summary” under “Stats Systems View”, and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.)
Answer: Again fopcussing on large model
ampere_sgemm_128x64_tn with total time onf 26 ms. This kernel is invoked 253 times during a single forward pass. 
If we consider both forward and backward passes, 
void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)2>>(int, T2, T3)
takes most time with total time of 125.827 ms. It is invoked 2108 times during a single  pass.

c) Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?
Elementwise math / tensor ops (memory-bound, lots of launches)

3.5% — void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
Does: elementwise multiply (x * y). Common for scaling/masking/gates.

1.7% — void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() const::[lambda() (instance 7)]::operator ()() const::[lambda(float) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
Does: copy / type / layout move (TensorIterator-based copy). Often from contiguous/cast/materialization.

1.3% — void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::array<char *, (unsigned long)3>>(int, T2, T3)
Does: elementwise add (x + y [+ maybe z]) vectorized. Often residual adds / bias adds.

1.1% — void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3)
Does: vectorized elementwise multiply (often fused/broadcasted variant).

Reductions (softmax + normalization stats)
1.0% — void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, (int)4>>(T3)
Does: mean reduction (compute averages). Often LayerNorm/RMSNorm stats.

d) Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you’d do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?
Answer:
Profiler results: [Profiler Results](cs336_systems/outputs/profiler_results/)
Fraction of time spent on matrix multiplication changes 85.3% for forward pass to 18.4% in backward pass. Did this analysis for large model with context length 32 on L4 gpu.
Breakdown (forward and backward pass):
| Category                                       |     % time | What's included                                     |
| ---------------------------------------------- | ---------: |                                                     |
| **Elementwise / pointwise**                    | **~80.1%** | vectorized `Mul`, `Add`, `Div`, `sqrt` etc|
| **Matrix multiply (GEMM / matmul)**            | **~18.4%** | `ampere_sgemm_*` + `cutlass` + `cublasLt`           |
| **Reductions**                                 |  **~0.4%** | `reduce_kernel` for `sum` + `mean` + `max`          |
| **Indexing / scatter-gather**                  |  **~0.4%** | `index_elementwise_kernel`, `index_put_kernel_impl`,|
| **Sort / misc (small)**                        |  **~0.1%** | `DeviceRadixSortSingleTileKernel`                   |
| **Other tiny kernels (rounding error / tail)** |  **~0.6%** | remaining (sigmoid, masked_fill etc.)               |
   

Breakdown (forward pass only):
| Category                            |     % time | What’s included                                                |
| **Matrix multiply (GEMM / matmul)** | **~85.3%** | `ampere_sgemm_128x64_tn`, `ampere_sgemm_128x128_nn`, `cublasLt`|
| **Elementwise / pointwise**         |  **~9.4%** | vectorized `Mul`, `Add`, `Div`, `sqrt` etc                     |
| **Indexing / scatter-gather**       |  **~3.3%** | `index_elementwise_kernel` + `index_put`                       |
| **Copies / data movement**          |  **~2.6%** | `direct_copy_kernel_cuda` + unrolled copy kernels              |
| **Reductions**                      |  **~1.9%** | `reduce mean` + `reduce max` + `reduce sum`                    |
| **Other / misc tiny**               |  **~0.6%** | `triu_tril`, `arange`, small glue kernels etc.                 |


e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?
Answer:
QK matmul: 50.765 ms 
Softmax: 43.047 ms 
AV matmul: 22.014 ms

Softmax percentage of total time: 
43.047/(50.765+22.014+43.047) ≈37%
Matmul percentage of total time:≈63%

H = 20
L = 32
D = 1280
QKᵀ FLOPs = 2xL²×D
AV FLOPs = 2xL²×D
So total matmul FLOPs for one layer and batch 1 = 4xL²×D = 4x32²x1280 = 5242880 FLOPs
Softmax FLOPs -> Shape of matrix: HxLxL -> with 5 ops (max, sum, neg, exp, div) -> 5xHxL² = 
= 5x20x32² = 102400 FLOPs

Softmax percentage of total FLOPs = 102400 / (5242880 + 102400) ≈ 1.9%
 
Reason: softmax is dominated by memory traffic + reductions + exp/div, which have low arithmetic intensity and lower hardware throughput, while matmuls run on extremely optimized GEMM kernels with high TFLOPs/s. Therefore, softmax is “cheap in FLOPs” but “expensive in time.”
