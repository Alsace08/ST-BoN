import os
import sys
import re
import json
from collections import Counter

from tqdm import tqdm

project_root_path = os.environ["PROJECT_PATH"]
sys.path.append(project_root_path)

from Data.load_data import DatasetInfo
from prompt import *
from decoders import *
from support import *


class Utility:
    def __init__(self, global_config):
        self.global_config = global_config

        self.model_name = global_config["model_name"]
        self.dataset_name = global_config["dataset_name"]
        self.decoding_mode = global_config["decoding_mode"]
        self.tokenizer = global_config["tokenizer"]

        self.top_k = global_config.get("top_k", None)
        self.top_p = global_config.get("top_p", None)
        self.temperature_t = global_config.get("temperature_t", None)
        self.num_sample = global_config.get("num_sample", None)
        self.tau_coeff = global_config.get("tau_coeff", None)


    def parse_input(self, sample):
        raw_input = sample["question"]
        raw_output = sample["answer"]

        model_input_text = DATASET_PROMPTS[self.dataset_name].replace("{raw_input}", raw_input)
        if self.dataset_name == "theoremqa":
            model_input_text = model_input_text.replace("{answer_type}", sample["answer_type"])

        model_input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": model_input_text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        model_input_len = len(model_input_ids[0])

        print(f"********** Input Text (length: {model_input_len}) **********\n{raw_input}\n")
        return raw_input, raw_output, {"text": model_input_text, "ids": model_input_ids}

    def parse_output(self, text):
        pattern = re.compile(r"\\boxed{")
        matches = pattern.finditer(text)
        results = []

        for match in matches:
            start_pos = match.end()
            brace_count = 1
            i = start_pos

            while i < len(text) and brace_count > 0:
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                i += 1

            if brace_count == 0:
                results.append(text[start_pos:i - 1])

        return results

    def _build_sampling_subdir(self):
        """
        Build parameter subdir for decoding modes that depend on sampling / geometric params.
        """
        if self.decoding_mode in ["sc", "st-bon"]:
            sampling_params = (
                f"k{self.top_k}"
                f"_p{self.top_p}"
                f"_t{self.temperature_t}"
            )

            if self.decoding_mode in ["sc", "st-bon"]:
                sampling_params += f"_n{self.num_sample}"

            if self.decoding_mode == "st-bon":
                sampling_params += f"_tau{self.tau_coeff}"

            return sampling_params

        return None

    def save_output(self, all_output):
        filedir = os.path.join(
            project_root_path,
            "Output",
            self.model_name,
            self.dataset_name,
            self.decoding_mode,
        )

        sampling_subdir = self._build_sampling_subdir()
        if sampling_subdir is not None:
            filedir = os.path.join(filedir, sampling_subdir)

        os.makedirs(filedir, exist_ok=True)

        filepath = os.path.join(filedir, f"{all_output['id']}.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(all_output, f, ensure_ascii=False, indent=2)

    def print_full_output(self, result):
        """
        Print full generation text to terminal.
        """
        if self.decoding_mode == "sc":
            print("********** Full Generated Texts **********")
            for idx, text in enumerate(result["output_text"]):
                print(f"[sample {idx}]")
                print(text)
                print("-" * 80)
            print()
        else:
            print("********** Full Generated Text **********")
            print(result["output_text"])
            print()


class InferenceRunner:
    def __init__(self, global_config: dict):
        self.global_config = global_config

        self.dataset_name = global_config["dataset_name"]
        self.data_size = global_config["data_size"]
        self.decoding_mode = global_config["decoding_mode"]

        self.data_loader = DatasetInfo(self.dataset_name)
        self.data_all = self.data_loader.data

        self.utils = Utility(global_config)

        self.greedy_decoder = GreedyDecoder(global_config)
        self.sc_decoder = SelfConsistencyDecoder(global_config)
        self.stbon_decoder = STBoNDecoder(global_config)


    def run_one_sample(self, sample, sample_id):
        raw_input, raw_output, model_input = self.utils.parse_input(sample)
        model_input_ids = model_input["ids"]

        if self.decoding_mode == "greedy":
            print("********** Decoding with greedy **********")
            result = self.greedy_decoder.decode(model_input_ids)
            predicted_answer = self.utils.parse_output(result["output_text"])

        elif self.decoding_mode == "sc":
            print("********** Decoding with self-consistency **********")
            result = self.sc_decoder.decode(model_input_ids)

            parsed_answers = []
            for text in result["output_text"]:
                parsed = self.utils.parse_output(text)
                if isinstance(parsed, list):
                    if len(parsed) > 0:
                        parsed_answers.append(parsed[0])
                elif parsed is not None:
                    parsed_answers.append(parsed)

            if len(parsed_answers) == 0:
                predicted_answer = []
            else:
                counter = Counter(parsed_answers)
                max_count = max(counter.values())
                predicted_answer = sorted([ans for ans, cnt in counter.items() if cnt == max_count])

            result["meta"]["sampled_answers"] = parsed_answers
            result["meta"]["top_answers"] = predicted_answer
            result["meta"]["top_answer_count"] = len(predicted_answer)

        elif self.decoding_mode == "st-bon":
            print("********** Decoding with st-bon **********")
            result = self.stbon_decoder.decode(model_input_ids)
            predicted_answer = self.utils.parse_output(result["output_text"])

        else:
            raise ValueError(f"Unsupported decoding_mode: {self.decoding_mode}")

        all_output = {
            "id": sample_id,
            "true_answer": raw_output,
            "predicted_answer": predicted_answer,
            "answer_type": sample["answer_type"] if self.dataset_name == "theoremqa" else "",
            "question": raw_input,
            "meta": result["meta"],
        }

        # self.utils.print_full_output(result)

        print(f"********** Predicted Answer **********\n{predicted_answer}\n")
        print(f"********** Ground-truth Text **********\n{raw_output}\n")
        print(f"********** Meta **********\n{result['meta']}\n")

        return all_output

    def dataset_inference(self):
        for i in tqdm(range(self.data_size)):
            print("*" * 30 + f" index {str(i)} " + "*" * 30)
            sample = self.data_all[i]

            all_output = self.run_one_sample(sample, i)
            self.utils.save_output(all_output)