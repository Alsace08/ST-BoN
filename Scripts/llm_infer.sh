#!/bin/bash
set -euo pipefail

export PROJECT_PATH="/your/path/to/ST-BoN"
export CUDA_VISIBLE_DEVICES="0,1"

model_name="Qwen-2.5-7B-Instruct"

decoding_mode="greedy"
# decoding_mode="sc"
# decoding_mode="st-bon"

dataset_list=(MATH-500 MMLU theoremqa)

max_output_token=4096

# =========================
# shared sampling params
# =========================
top_k=20
top_p=0.95
temperature_t=0.7
num_sample=20

# =========================
# st-bon
# =========================
tau_coeff=1


for dataset in "${dataset_list[@]}"; do
    cmd=(
        python main.py
        --model_name "$model_name"
        --dataset "$dataset"
        --decoding_mode "$decoding_mode"
        --max_output_token "$max_output_token"
    )

    if [ "$decoding_mode" = "sc" ]; then
        cmd+=(
            --top_k "$top_k"
            --top_p "$top_p"
            --temperature_t "$temperature_t"
            --num_sample "$num_sample"
        )
    elif [ "$decoding_mode" = "st-bon" ]; then
        cmd+=(
            --top_k "$top_k"
            --top_p "$top_p"
            --temperature_t "$temperature_t"
            --num_sample "$num_sample"
            --tau_coeff "$tau_coeff"
        )
    fi


    echo "Running:"
    printf ' %q' "${cmd[@]}"
    echo

    "${cmd[@]}"
done