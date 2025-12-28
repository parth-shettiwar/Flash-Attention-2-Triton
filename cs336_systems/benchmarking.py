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

def benchmark_model(module_model, hyperparameters, vocab_size, batch_size, context_length, forward_only=False):
    """
    Benchmark the model for n_steps steps.
    """
    n_steps = 10
    warmup_steps = 5
    model = module_model(vocab_size, context_length, **hyperparameters, rope_theta=10000.0)
    # generate random data
    data = torch.randint(0, vocab_size, (batch_size, context_length))

    
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
    context_length = 8

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
    for model_size, hyperparameters in hyperparameters.items():
        print("Starting benchmark for model size: ", model_size)
        forward_time, backward_time = benchmark_model(BasicsTransformerLM, hyperparameters, vocab_size, batch_size, context_length, forward_only=False)
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
    
    with open("cs336_systems/outputs/benchmark_timings.json", "w") as f:
        json.dump(dic_timings, f)