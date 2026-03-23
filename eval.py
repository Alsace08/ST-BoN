import os
import re
import json
import argparse
from typing import Any, Dict, List

from support import (
    SUPPORTED_DATASETS,
    SUPPORTED_DECODING_MODES,
    SUPPORTED_MODELS,
)
from match import math_equal


def infer_dataset_from_folder(folder: str) -> str:
    folder_norm = os.path.normpath(folder)
    parts = folder_norm.split(os.sep)

    for part in parts:
        if part in SUPPORTED_DATASETS:
            return part

    lower_parts = [p.lower() for p in parts]
    dataset_map = {d.lower(): d for d in SUPPORTED_DATASETS}
    for p in lower_parts:
        if p in dataset_map:
            return dataset_map[p]

    folder_lower = folder_norm.lower()
    for d in SUPPORTED_DATASETS:
        if d.lower() in folder_lower:
            return d

    raise ValueError(
        f"Cannot infer dataset from folder path: {folder}\n"
        f"Supported datasets: {SUPPORTED_DATASETS}"
    )


def infer_decoding_mode_from_folder(folder: str) -> str:
    folder_norm = os.path.normpath(folder)
    parts = folder_norm.split(os.sep)

    for part in parts:
        if part in SUPPORTED_DECODING_MODES:
            return part

    lower_parts = [p.lower() for p in parts]
    mode_map = {m.lower(): m for m in SUPPORTED_DECODING_MODES}
    for p in lower_parts:
        if p in mode_map:
            return mode_map[p]

    folder_lower = folder_norm.lower()
    for m in SUPPORTED_DECODING_MODES:
        if m.lower() in folder_lower:
            return m

    raise ValueError(
        f"Cannot infer decoding mode from folder path: {folder}\n"
        f"Supported decoding modes: {SUPPORTED_DECODING_MODES}"
    )


def infer_model_name_from_folder(folder: str) -> str:
    folder_norm = os.path.normpath(folder)
    parts = folder_norm.split(os.sep)

    for part in parts:
        if part in SUPPORTED_MODELS:
            return part

    lower_parts = [p.lower() for p in parts]
    model_map = {m.lower(): m for m in SUPPORTED_MODELS}
    for p in lower_parts:
        if p in model_map:
            return model_map[p]

    folder_lower = folder_norm.lower()
    for m in SUPPORTED_MODELS:
        if m.lower() in folder_lower:
            return m

    raise ValueError(
        f"Cannot infer model name from folder path: {folder}\n"
        f"Supported models: {SUPPORTED_MODELS}"
    )


def infer_sampling_params_from_folder(folder: str, decoding_mode: str) -> Dict[str, Any]:
    """
    Expected path suffix examples:
      greedy:
        .../greedy/
      sc:
        .../sc/k20_p0.95_t0.7_n10
      st-bon:
        .../st-bon/k20_p0.95_t0.7_n10_tau1.0
    """
    folder_norm = os.path.normpath(folder)
    parts = folder_norm.split(os.sep)

    candidate = parts[-1]
    if not re.search(r"k\d+", candidate):
        if len(parts) >= 2 and re.search(r"k\d+", parts[-2]):
            candidate = parts[-2]

    params = {
        "top_k": None,
        "top_p": None,
        "temperature_t": None,
        "num_sample": None,
        "tau_coeff": None,
    }

    mk = re.search(r"k([0-9]+)", candidate)
    mp = re.search(r"_p([0-9]*\.?[0-9]+)", candidate)
    mt = re.search(r"_t([0-9]*\.?[0-9]+)", candidate)
    mn = re.search(r"_n([0-9]+)", candidate)
    mtau = re.search(r"_tau([0-9]*\.?[0-9]+)", candidate)

    if mk:
        params["top_k"] = int(mk.group(1))
    if mp:
        params["top_p"] = float(mp.group(1))
    if mt:
        params["temperature_t"] = float(mt.group(1))
    if mn:
        params["num_sample"] = int(mn.group(1))
    if mtau:
        params["tau_coeff"] = float(mtau.group(1))

    if decoding_mode == "greedy":
        return params

    if decoding_mode == "sc":
        required = ["top_k", "top_p", "temperature_t", "num_sample"]
        missing = [k for k in required if params[k] is None]
        if missing:
            raise ValueError(
                f"Cannot infer sampling params for decoding_mode='sc' from folder: {folder}\n"
                f"Missing fields: {missing}"
            )
        return params

    if decoding_mode == "st-bon":
        required = ["top_k", "top_p", "temperature_t", "num_sample", "tau_coeff"]
        missing = [k for k in required if params[k] is None]
        if missing:
            raise ValueError(
                f"Cannot infer sampling params for decoding_mode='st-bon' from folder: {folder}\n"
                f"Missing fields: {missing}"
            )
        return params

    raise ValueError(f"Unsupported decoding mode: {decoding_mode}")


def load_result_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_result_files(folder: str) -> List[str]:
    files = []
    for name in os.listdir(folder):
        if name.endswith(".json") or name.endswith(".jsonl"):
            files.append(os.path.join(folder, name))
    return sorted(files)


def evaluate_folder(folder: str, dataset: str, model_name: str, decoding_mode: str, sampling_params: Dict[str, Any]) -> Dict[str, Any]:
    files = collect_result_files(folder)
    if len(files) == 0:
        raise ValueError(f"No json/jsonl result files found in folder: {folder}")

    total = 0
    correct = 0

    total_inference_time = 0.0
    inference_time_count = 0

    total_output_length = 0.0
    output_length_count = 0

    details = []

    for path in files:
        record = load_result_file(path)

        true_answer = record.get("true_answer")
        pred_answer = record.get("predicted_answer")
        answer_type = record.get("answer_type", "")
        meta = record.get("meta", {}) or {}

        if isinstance(pred_answer, list):
            pred_answer = None if len(pred_answer) == 0 else pred_answer[0]

        binary = math_equal(pred_answer, true_answer)
        print(pred_answer, true_answer, binary)

        inference_time = meta.get("inference_time", None)
        if inference_time is not None:
            try:
                total_inference_time += float(inference_time)
                inference_time_count += 1
            except Exception:
                pass

        output_length = meta.get("output_length", None)
        if output_length is not None:
            try:
                total_output_length += float(output_length)
                output_length_count += 1
            except Exception:
                pass

        total += 1
        correct += int(binary)

        details.append({
            "id": record.get("id"),
            "file": os.path.basename(path),
            "true_answer": true_answer,
            "predicted_answer": pred_answer,
            "answer_type": answer_type,
            "correct": bool(binary),
            "inference_time": inference_time,
            "output_length": output_length,
        })

    accuracy = correct / total if total > 0 else 0.0
    avg_inference_time = total_inference_time / inference_time_count if inference_time_count > 0 else None
    avg_output_length = total_output_length / output_length_count if output_length_count > 0 else None

    return {
        "folder": folder,
        "model_name": model_name,
        "dataset": dataset,
        "decoding_mode": decoding_mode,
        "top_k": sampling_params.get("top_k"),
        "top_p": sampling_params.get("top_p"),
        "temperature_t": sampling_params.get("temperature_t"),
        "num_sample": sampling_params.get("num_sample"),
        "tau_coeff": sampling_params.get("tau_coeff"),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_inference_time": avg_inference_time,
        "avg_output_length": avg_output_length,
        "details": details,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=str, required=True, help="Folder containing result json/jsonl files")
    ap.add_argument("--save_detail", action="store_true", help="Whether to save per-sample details")
    ap.add_argument("--detail_path", type=str, default=None, help="Path to save detail json")
    args = ap.parse_args()

    dataset = infer_dataset_from_folder(args.folder)
    decoding_mode = infer_decoding_mode_from_folder(args.folder)
    model_name = infer_model_name_from_folder(args.folder)
    sampling_params = infer_sampling_params_from_folder(args.folder, decoding_mode)

    result = evaluate_folder(
        folder=args.folder,
        dataset=dataset,
        model_name=model_name,
        decoding_mode=decoding_mode,
        sampling_params=sampling_params,
    )

    print("=" * 60)
    print(f"Model                   : {result['model_name']}")
    print(f"Dataset                 : {result['dataset']}")
    print(f"Decoding Mode           : {result['decoding_mode']}")
    print("=" * 60)
    print(f"top_k                   : {result['top_k']}")
    print(f"top_p                   : {result['top_p']}")
    print(f"temperature_t           : {result['temperature_t']}")
    print(f"num_sample              : {result['num_sample']}")
    print(f"tau_coeff               : {result['tau_coeff']}")
    print("=" * 60)
    print(f"Total                   : {result['total']}")
    print(f"Correct                 : {result['correct']}")
    print(f"Accuracy                : {(result['accuracy'] * 100):.2f}")
    if result["avg_inference_time"] is None:
        print("Avg Inference Time      : None")
    else:
        print(f"Avg Inference Time      : {round(result['avg_inference_time'], 4)}s")

    if result["avg_output_length"] is None:
        print("Avg Output Length       : None")
    else:
        print(f"Avg Output Length       : {round(result['avg_output_length'], 2)}")

    print("=" * 60)

    if args.save_detail:
        detail_path = args.detail_path
        if detail_path is None:
            detail_path = os.path.join(args.folder, "eval_detail.json")
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved detail to: {detail_path}")


if __name__ == "__main__":
    main()