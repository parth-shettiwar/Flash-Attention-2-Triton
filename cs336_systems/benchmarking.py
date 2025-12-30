import functools
import json
import logging
import math
import os
import sys
from einops import rearrange, einsum
import einx

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int


import timeit
import time
import matplotlib.pyplot as plt
import numpy as np
import torch.cuda.nvtx as nvtx

import argparse
import gc
import importlib.util
import importlib.machinery

USE_MY_IMPLEMENTATION = True
if not USE_MY_IMPLEMENTATION:
    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.optimizer import AdamW
else:
    # Load BPETokenizerTransformer/cs336_basics as a package explicitly to avoid conflicts
    pkg_root = "/teamspace/studios/this_studio/Flash-Attention-2-Triton/BPETokenizerTransformer/cs336_basics"
    if "cs336_basics" not in sys.modules:
        pkg_spec = importlib.machinery.ModuleSpec("cs336_basics", loader=None, is_package=True)
        pkg_module = importlib.util.module_from_spec(pkg_spec)
        pkg_module.__path__ = [pkg_root]
        sys.modules["cs336_basics"] = pkg_module

    # Load transformer
    spec_t = importlib.util.spec_from_file_location("cs336_basics.transformer", f"{pkg_root}/transformer.py")
    mod_t = importlib.util.module_from_spec(spec_t)
    spec_t.loader.exec_module(mod_t)
    sys.modules["cs336_basics.transformer"] = mod_t
    mymodel = mod_t.TransformerLM

    # Load optimizer
    spec_o = importlib.util.spec_from_file_location("cs336_basics.train_transformer", f"{pkg_root}/train_transformer.py")
    mod_o = importlib.util.module_from_spec(spec_o)
    spec_o.loader.exec_module(mod_o)
    sys.modules["cs336_basics.train_transformer"] = mod_o
    myoptimizer = mod_o.AdamW

def benchmark_model(module_model, hyperparameters, vocab_size, batch_size, context_length, forward_only=False, warmup_exp=False, device="cuda"):
    """
    Benchmark the model for n_steps steps.
    """
    n_steps = 10
    warmup_steps = 5 if warmup_exp else 0
    model = module_model(vocab_size, context_length, **hyperparameters, rope_theta=10000.0)
    # generate random data
    data = torch.randint(0, vocab_size, (batch_size, context_length))
    if device == "cuda":
        data = data.to(device)
        model = model.to(device)
    
    for _ in range(warmup_steps):
        loss = model(data).mean()
        if not forward_only:
            loss.backward()
            model.zero_grad() 

    # timeit.default_timer()
    forward_times = []
    backward_times = []
    for _ in range(n_steps):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = timeit.default_timer()
        loss = model(data).mean()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = timeit.default_timer() 
        forward_times.append(end_time - start_time)  
   
        if not forward_only:
            if torch.cuda.is_available():
                torch.cuda.synchronize()  
            start_time_backward = timeit.default_timer()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time_backward = timeit.default_timer()
            model.zero_grad() 
            backward_times.append(end_time_backward - start_time_backward)

    # clear cuda memory
    del model, data, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return forward_times, backward_times

def profile_model(module_model, model_name, hyperparameters, vocab_size, batch_size, context_length, forward_only=False, warmup_exp=True, device="cuda", use_myimplementation=False):
    """
    Profile the model for n_steps steps.
    
    Run with: nsys profile -o profile_<model_name> --capture-range=cudaProfilerApi \
              .venv/bin/python cs336_systems/benchmarking.py
    """
    n_steps = 10
    model = module_model(vocab_size, context_length, **hyperparameters, rope_theta=10000.0)
    data = torch.randint(0, vocab_size, (batch_size, context_length))
    if use_myimplementation:
        optimizer = myoptimizer(model.parameters(), lr=1e-4)
    else:
        optimizer = AdamW(model.parameters(), lr=1e-4)
    if device == "cuda":
        data = data.to(device)
        model = model.to(device)
    warmup_steps = 5 if warmup_exp else 0
    for _ in range(warmup_steps):
        loss = model(data).mean()
        if not forward_only:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    torch.cuda.synchronize()
    
    # Start CUDA profiler
    torch.cuda.cudart().cudaProfilerStart()

    for step in range(n_steps):
        nvtx.range_push(f"Step {step}")
        
        nvtx.range_push("Forward")
        loss = model(data).mean()
        torch.cuda.synchronize()
        nvtx.range_pop()

        if not forward_only:
            nvtx.range_push("Backward")
            loss.backward()
            torch.cuda.synchronize()
            nvtx.range_pop()    

        if not forward_only:
            nvtx.range_push("Optimizer Step")
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            nvtx.range_pop()
        
        nvtx.range_pop()  # End step

    torch.cuda.cudart().cudaProfilerStop()

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
        # "2.7B": {
        #     "d_model": 2560,
        #     "d_ff": 10240,
        #     "num_layers": 32,
        #     "num_heads": 32,
        # },
    }

    # benchmark the model
    # dic_timings = {}
    # warmup_exp = False
    # for model_size, hyperparameters in hyperparameters.items():
    #     print("Starting benchmark for model size: ", model_size)
    #     forward_time, backward_time = benchmark_model(BasicsTransformerLM, hyperparameters, vocab_size, batch_size, context_length, forward_only=False, warmup_exp=warmup_exp)
    #     dic_timings[model_size] = {
    #         "forward_time": {
    #             "avg_times": sum(forward_time) / len(forward_time),
    #             "std_times": torch.std(torch.tensor(forward_time)).item(),
    #         },
    #         "backward_time": {
    #             "avg_times": sum(backward_time) / len(backward_time),
    #             "std_times": torch.std(torch.tensor(backward_time)).item(),
    #         },
    #     }
    #     print("Benchmark for model size: ", model_size, " completed")
    #     print("Forward time: ", forward_time, " seconds")
    #     print("Backward time: ", backward_time, " seconds")
    #     if torch.cuda.is_available():
    #         gc.collect()
    #         torch.cuda.empty_cache()

    # base_path = "cs336_systems/outputs_l4"
    # file_name = "benchmark_timings.json" if warmup_exp else "benchmark_timings_no_warmup.json"
    # with open(os.path.join(base_path, file_name), "w") as f:
    #     json.dump(dic_timings, f, indent=4)
    # # read the json file and plot
    # with open(os.path.join(base_path, file_name), "r") as f:
    #     dic_timings = json.load(f)
    # for model_size, timings in dic_timings.items():
    #     print(model_size)
    #     print(timings)
    #     print("Forward time: ", timings["forward_time"]["avg_times"], " seconds")
    #     print("Backward time: ", timings["backward_time"]["avg_times"], " seconds")
    #     print("Forward time std: ", timings["forward_time"]["std_times"], " seconds")
    #     print("Backward time std: ", timings["backward_time"]["std_times"], " seconds")
    #     print("--------------------------------")
    # # plot the forward and backward times for each model size
    # plt.figure(figsize=(10, 5))
    # plt.plot(dic_timings.keys(), [timings["forward_time"]["avg_times"] for timings in dic_timings.values()], label="Forward time")
    # plt.plot(dic_timings.keys(), [timings["backward_time"]["avg_times"] for timings in dic_timings.values()], label="Backward time")
    # plt.legend()
    # # save the plot
    # plt.savefig(os.path.join(base_path, file_name.replace(".json", ".png")))



    # profile the model
    # capture command line arguement for model size
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="small")
    # parser.add_argument("--forward_only", action="store_true")
    # args = parser.parse_args()
    # model_size = args.model
    # forward_only = args.forward_only
    # hyperparameters = hyperparameters[model_size]
    # print("Starting profiling for model size: ", model_size)
    # profile_model(BasicsTransformerLM, model_size, hyperparameters, vocab_size, batch_size, context_length, forward_only=forward_only, warmup_exp=True)
    # print("Profiling for model size: ", model_size, " completed")
    # print("--------------------------------")

    # do on mymodel
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="small")
    parser.add_argument("--forward_only", action="store_true")
    args = parser.parse_args()
    model_size = args.model
    forward_only = args.forward_only
    hyperparameters = hyperparameters[model_size]
    print("Starting profiling for model size: ", model_size)
    profile_model(mymodel, model_size, hyperparameters, vocab_size, batch_size, context_length, forward_only=forward_only, warmup_exp=True, use_myimplementation=USE_MY_IMPLEMENTATION)
    print("Profiling for model size: ", model_size, " completed")
    print("--------------------------------")
