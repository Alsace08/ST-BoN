import os
import sys
import time
import math
import json
import random
import argparse
from tqdm import tqdm

import numpy as np
import pickle
import scipy.spatial
import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

project_root_path = os.environ["PROJECT_PATH"]
sys.path.append(project_root_path)

from arguments import arg_parses
from Data.load_data import DatasetInfo
from Model.load_model import load_base_model
from inference import InferenceRunner


if __name__ == '__main__':
    args = arg_parses()
    model, tokenizer, model_config = load_base_model(args)

    if args.print_model_parameter:
        print("********** Module Name and Size **********\n")
        for param_tensor in model.state_dict():
            print(param_tensor, '\t', model.state_dict()[param_tensor].size())

    global_config = {
        # model
        "model_name": args.model_name,
        "model_ckpt": model,
        "tokenizer": tokenizer,
        "model_config": model_config,
        "generation_config": GenerationConfig(),

        # dataset
        "dataset_name": args.dataset,
        "data_size": DatasetInfo(args.dataset).data_size,

        # decoding
        "decoding_mode": args.decoding_mode,
        "max_output_token": args.max_output_token,

        # sampling (sc / st-bon)
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature_t": args.temperature_t,
        "num_sample": args.num_sample,

        # st-bon
        "tau_coeff": args.tau_coeff,
    }

    print(f"***** Model Name: *****\n{args.model_name}")
    print(f"***** Dataset Name: *****\n{args.dataset}")
    print(f"***** Dataset Size: *****\n{global_config['data_size']}")
    print(f"***** Decoding Mode: *****\n{args.decoding_mode}")

    if args.decoding_mode in ["sc", "st-bon"]:
        print("***** Sampling / Geometric Params: *****")
        print(f"top_k = {args.top_k}")
        print(f"top_p = {args.top_p}")
        print(f"temperature_t = {args.temperature_t}")

    if args.decoding_mode == "sc":
        print(f"num_sample = {args.num_sample}")

    if args.decoding_mode == "st-bon":
        print(f"num_sample = {args.num_sample}")
        print(f"tau_coeff = {args.tau_coeff}")

    Infer = InferenceRunner(global_config)
    Infer.dataset_inference()
