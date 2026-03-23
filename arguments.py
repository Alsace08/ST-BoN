import argparse
from support import (
    SUPPORTED_DATASETS,
    SUPPORTED_DECODING_MODES,
    SUPPORTED_MODELS,
)


def arg_parses():
    parser = argparse.ArgumentParser(description="Self-Truncation-Best-of-N")

    general_group = parser.add_argument_group("General Parameters")
    general_group.add_argument("--model_name", type=str, default="Qwen2.5-7B-Instruct", choices=SUPPORTED_MODELS)
    general_group.add_argument("--dataset", type=str, default="mgsm", choices=SUPPORTED_DATASETS)
    general_group.add_argument("--max_output_token", type=int, default=4096)
    general_group.add_argument("--decoding_mode", type=str, default="greedy", choices=SUPPORTED_DECODING_MODES)
    general_group.add_argument("--print_model_parameter", action="store_true")

    sampling_group = parser.add_argument_group("Sampling Parameters")
    sampling_group.add_argument("--top_k", type=int, default=20)
    sampling_group.add_argument("--top_p", type=float, default=0.95)
    sampling_group.add_argument("--temperature_t", type=float, default=0.7)
    sampling_group.add_argument("--num_sample", type=int, default=10)

    stbon_group = parser.add_argument_group("ST-BON Specific Parameters")
    stbon_group.add_argument("--tau_coeff", type=float, default=1.0)

    args = parser.parse_args()
    return args