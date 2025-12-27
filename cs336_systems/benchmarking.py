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

def benchmark_model(model, data, n_steps, warmup_steps, hyperparameters, forward_only=False):
    """
    Benchmark the model for n_steps steps.
    """
    model = BasicsTransformerLM(**hyperparameters)
    for _ in range(warmup_steps):
        model(data)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # timeit.default_timer()
    times = []
    for _ in range(n_steps):
        start_time = timeit.default_timer()
        out = model(data)
        if not forward_only:
            out.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = timeit.default_timer()   
        times.append(end_time - start_time)

    mean_time = sum(times) / n_steps

    return mean_time