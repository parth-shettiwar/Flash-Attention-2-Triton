import functools
import json
import logging
import math
import os
from einops import rearrange, einsum
import einx

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int
from cs336_basics.model import BasicsTransformerLM

import timeit
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda.nvtx as nvtx

def benchmark_model(module_model, hyperparameters, vocab_size, batch_size, context_length, forward_only=False, warmup_exp=False, device="cuda"):
    """
    Benchmark the model for n_steps steps.
    """
    n_steps = 1
    warmup_steps = 5 if warmup_exp else 0
    model = module_model(vocab_size, context_length, **hyperparameters, rope_theta=10000.0)
    # generate random data
    data = torch.randint(0, vocab_size, (batch_size, context_length))
    if device == "cuda":
        data = data.to(device)
        model = model.to(device)
    
    for _ in range(warmup_steps):
        model(data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # timeit.default_timer()
    forward_times = []
    for _ in range(n_steps):
        start_time = timeit.default_timer()
        model(data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = timeit.default_timer()   
        forward_times.append(end_time - start_time)

    backward_times = []
    if not forward_only:
        for _ in range(warmup_steps):
            out = model(data)
            out.mean().backward()

        for _ in range(n_steps):
            out = model(data)
            back_out = out.mean()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = timeit.default_timer()
            back_out.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = timeit.default_timer()
            backward_times.append(end_time - start_time)

    return forward_times, backward_times

if __name__ == "__main__":
    vocab_size = 10000
    batch_size = 4
    context_length = 32

    hyperparameters ={
        "small": {
            "d_model": 768,
            "d_ff": 3072,
            "num_layers": 12,
            "num_heads": 12,
        },
        "medium": {
            "d_model": 1024,
            "d_ff": 4096,
            "num_layers": 24,
            "num_heads": 16,
            },
        "large": {
            "d_model": 1280,
            "d_ff": 5120,
            "num_layers": 36,
            "num_heads": 20,
        },
        "xl": {
            "d_model": 1600,
            "d_ff": 6400,
            "num_layers": 48,
            "num_heads": 25,
        },
        "2.7B": {
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32,
        },
    }
    dic_timings = {}
    warmup_exp = False
    for model_size, hyperparameters in hyperparameters.items():
        print("Starting benchmark for model size: ", model_size)
        forward_time, backward_time = benchmark_model(BasicsTransformerLM, hyperparameters, vocab_size, batch_size, context_length, forward_only=False, warmup_exp=warmup_exp)
        dic_timings[model_size] = {
            "forward_time": {
                "avg_times": sum(forward_time) / len(forward_time),
                "std_times": torch.std(torch.tensor(forward_time)).item(),
            },
            "backward_time": {
                "avg_times": sum(backward_time) / len(backward_time),
                "std_times": torch.std(torch.tensor(backward_time)).item(),
            },
        }
        print("Benchmark for model size: ", model_size, " completed")
        print("Forward time: ", forward_time, " seconds")
        print("Backward time: ", backward_time, " seconds")
    base_path = "cs336_systems/outputs"
    file_name = "benchmark_timings.json" if warmup_exp else "benchmark_timings_no_warmup.json"
    with open(os.path.join(base_path, file_name), "w") as f:
        json.dump(dic_timings, f, indent=4)
    # read the json file and plot
    with open(os.path.join(base_path, file_name), "r") as f:
        dic_timings = json.load(f)
    for model_size, timings in dic_timings.items():
        print(model_size)
        print(timings)
        print("Forward time: ", timings["forward_time"]["avg_times"], " seconds")
        print("Backward time: ", timings["backward_time"]["avg_times"], " seconds")
        print("Forward time std: ", timings["forward_time"]["std_times"], " seconds")
        print("Backward time std: ", timings["backward_time"]["std_times"], " seconds")
        print("--------------------------------")
    # plot the forward and backward times for each model size
    plt.figure(figsize=(10, 5))
    plt.plot(dic_timings.keys(), [timings["forward_time"]["avg_times"] for timings in dic_timings.values()], label="Forward time")
    plt.plot(dic_timings.keys(), [timings["backward_time"]["avg_times"] for timings in dic_timings.values()], label="Backward time")
    plt.legend()
    # save the plot
    plt.savefig(os.path.join(base_path, file_name.replace(".json", ".png")))