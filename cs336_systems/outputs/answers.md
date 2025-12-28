Problem (benchmarking_script)
Answer:
Refer: [Results](cs336_systems/outputs/benchmark_timings.json)
Timing plot: [Timing Plot](cs336_systems/outputs/benchmark_timings.png)
Overall very little std deviation, and model increase leads to more time taken for both forward and backward pass. 2.7B vs xl we observe that xl is slower than 2.7B. This maybe due to lesser num layers in 2.7B model.
If no warmup is done, the passes have both higher std deviation and higher average time as expected.
