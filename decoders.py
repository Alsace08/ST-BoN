import time
from collections import Counter

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trim_after_first_terminator(seq, terminators):
    out = []
    for x in seq:
        if x in terminators:
            break
        out.append(x)
    return out


class STBoNEarlyStopCriteria(StoppingCriteria):
    """
    Stop stage-1 sampling at c + tau, where tau = tau_coeff * c.
    """

    def __init__(self, prompt_len, eos_token_ids, tau_coeff):
        super().__init__()
        self.prompt_len = prompt_len
        self.eos_token_ids = set(int(x) for x in eos_token_ids if x is not None)
        self.tau_coeff = tau_coeff

        self.c = None
        self.tau = None
        self.stop_step = None

    def _all_prefixes_pairwise_distinct(self, generated_part: torch.Tensor) -> bool:
        uniq = torch.unique(generated_part, dim=0)
        return uniq.size(0) == generated_part.size(0)

    def __call__(self, input_ids, scores, **kwargs):
        cur_len = input_ids.size(1)
        gen_len = cur_len - self.prompt_len

        if gen_len <= 0:
            return False

        generated_part = input_ids[:, self.prompt_len:]

        if self.c is None:
            all_distinct = self._all_prefixes_pairwise_distinct(generated_part)
            last_tokens = generated_part[:, -1]
            any_eos = any(int(tok.item()) in self.eos_token_ids for tok in last_tokens)

            if all_distinct or any_eos:
                self.c = gen_len
                self.tau = max(1, int(self.tau_coeff * self.c))
                self.stop_step = self.c + self.tau

        if self.stop_step is not None and gen_len >= self.stop_step:
            return True

        return False


class CoECalculator:
    """
    For each sample i:
        F_i = scalar CoE feature

    Then:
        D(i,j) = (F_i - F_j)^2
        S_i = average_{j != i} D(i,j)

    The sample with the smallest S_i is the best sample at that step.
    """

    @staticmethod
    def _to_tensor(x, dev=None):
        if isinstance(x, torch.Tensor):
            t = x.detach().float()
        else:
            t = torch.tensor(x, dtype=torch.float32)
        if dev is not None:
            t = t.to(dev)
        return t

    @classmethod
    def compute_features_batch(cls, hs_all_layer_batch):
        """
        hs_all_layer_batch: [L, N, H]
        returns: [N]
        """
        hs = cls._to_tensor(
            hs_all_layer_batch,
            dev=hs_all_layer_batch.device if isinstance(hs_all_layer_batch, torch.Tensor) else None,
        )

        if hs.dim() != 3:
            raise ValueError(f"Expected [L, N, H], got shape {tuple(hs.shape)}")

        num_layers, num_samples, _ = hs.shape
        if num_layers <= 1:
            return torch.zeros(num_samples, dtype=torch.float32, device=hs.device)

        h0 = hs[0]
        hL = hs[-1]

        denom_M = torch.norm(h0 - hL, dim=-1).clamp_min(1e-8)

        cos_global = (h0 * hL).sum(dim=-1) / (
            torch.norm(h0, dim=-1).clamp_min(1e-8) *
            torch.norm(hL, dim=-1).clamp_min(1e-8)
        )
        cos_global = torch.clamp(cos_global, -1.0 + 1e-7, 1.0 - 1e-7)
        denom_A = torch.acos(cos_global).clamp_min(1e-8)

        h_cur = hs[:-1]
        h_nxt = hs[1:]

        M = torch.norm(h_cur - h_nxt, dim=-1)

        cos_local = (h_cur * h_nxt).sum(dim=-1) / (
            torch.norm(h_cur, dim=-1).clamp_min(1e-8) *
            torch.norm(h_nxt, dim=-1).clamp_min(1e-8)
        )
        cos_local = torch.clamp(cos_local, -1.0 + 1e-7, 1.0 - 1e-7)
        A = torch.acos(cos_local)

        vals = M / denom_M.unsqueeze(0) - A / denom_A.unsqueeze(0)
        feats = vals.mean(dim=0)
        return feats

    @staticmethod
    def compute_pairwise_scores(coe_features: torch.Tensor):
        feats = coe_features.float()
        n = feats.numel()

        if n <= 1:
            return torch.zeros_like(feats)

        diff = feats.unsqueeze(1) - feats.unsqueeze(0)
        dist = diff.pow(2)
        scores = (dist.sum(dim=1) - torch.diag(dist)) / (n - 1)
        return scores


class BaseDecoder:
    def __init__(self, global_config):
        self.global_config = global_config

        self.model = global_config["model_ckpt"]
        self.tokenizer = global_config["tokenizer"]
        self.generation_config = global_config["generation_config"]
        self.model_name = global_config["model_name"]
        self.max_output_token = global_config["max_output_token"]

        self.model.eval()

        self.runtime_device = self.get_runtime_device()
        self.terminators = self._build_terminators()
        self.terminator_set = set(self.terminators)
        self.pad_token_id = self.tokenizer.eos_token_id

    def get_runtime_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return device

    def _build_terminators(self):
        if "Llama" in self.model_name:
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        else:
            terminators = [self.tokenizer.eos_token_id]
        return [x for x in terminators if x is not None]

    def get_terminators(self):
        return self.terminators

    def extract_new_token_ids(self, sequence, prompt_len):
        token_ids = sequence[prompt_len:].tolist()
        token_ids = trim_after_first_terminator(token_ids, self.terminator_set)
        return token_ids

    def decode_single_sequence(self, sequence, prompt_len):
        token_ids = self.extract_new_token_ids(sequence, prompt_len)
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text, len(token_ids)

    def decode_multiple_sequences(self, sequences, prompt_len):
        output_texts = []
        output_lengths = []

        new_tokens = sequences[:, prompt_len:]
        for row in new_tokens:
            token_ids = trim_after_first_terminator(row.tolist(), self.terminator_set)
            output_texts.append(self.tokenizer.decode(token_ids, skip_special_tokens=True))
            output_lengths.append(len(token_ids))

        return output_texts, output_lengths


class GreedyDecoder(BaseDecoder):
    def __init__(self, global_config):
        super().__init__(global_config)

    @torch.inference_mode()
    def decode(self, input_ids):
        input_ids = input_ids.to(self.runtime_device)

        time_start = time.time()
        generation_output = self.model.generate(
            input_ids=input_ids,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.terminators,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            max_new_tokens=self.max_output_token,
            output_scores=True,
            do_sample=False,
        )
        time_end = time.time()

        inference_time = round(time_end - time_start, 4)
        prompt_len = input_ids.size(1)
        output_text, output_len = self.decode_single_sequence(
            generation_output.sequences[0],
            prompt_len,
        )

        meta = {
            "output_length": output_len,
            "inference_time": inference_time,
        }

        return {
            "output_text": output_text,
            "meta": meta,
        }


class SelfConsistencyDecoder(BaseDecoder):
    def __init__(self, global_config):
        super().__init__(global_config)

        self.top_k = global_config["top_k"]
        self.top_p = global_config["top_p"]
        self.temperature_t = global_config["temperature_t"]
        self.num_sample = global_config["num_sample"]

    @torch.inference_mode()
    def decode(self, input_ids):
        input_ids = input_ids.to(self.runtime_device)

        time_start = time.time()
        generation_output = self.model.generate(
            input_ids=input_ids,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.terminators,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            max_new_tokens=self.max_output_token,
            output_scores=True,
            do_sample=True,
            num_return_sequences=self.num_sample,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature_t,
        )
        time_end = time.time()

        inference_time = round(time_end - time_start, 4)
        prompt_len = input_ids.size(1)

        output_texts, output_lengths = self.decode_multiple_sequences(
            generation_output.sequences,
            prompt_len,
        )

        avg_output_length = float(sum(output_lengths) / len(output_lengths)) if len(output_lengths) > 0 else 0.0

        meta = {
            "output_length": avg_output_length,
            "inference_time": inference_time,
        }

        return {
            "output_text": output_texts,
            "meta": meta,
        }


class STBoNDecoder(BaseDecoder):
    def __init__(self, global_config):
        super().__init__(global_config)

        self.top_k = global_config["top_k"]
        self.top_p = global_config["top_p"]
        self.temperature_t = global_config["temperature_t"]
        self.num_sample = global_config["num_sample"]
        self.tau_coeff = global_config["tau_coeff"]
        self.stage2_on_gpu = global_config.get("stbon_stage2_on_gpu", True)

        self.coe_calculator = CoECalculator()

    @torch.inference_mode()
    def stage1_sample_until_stop(self, input_ids):
        prompt_len = input_ids.size(1)

        stopper = STBoNEarlyStopCriteria(
            prompt_len=prompt_len,
            eos_token_ids=self.terminators,
            tau_coeff=self.tau_coeff,
        )

        partial_output = self.model.generate(
            input_ids=input_ids.to(self.runtime_device),
            pad_token_id=self.pad_token_id,
            eos_token_id=self.terminators,
            generation_config=self.generation_config,
            return_dict_in_generate=True,
            max_new_tokens=self.max_output_token,
            output_hidden_states=True,
            output_scores=True,
            do_sample=True,
            num_return_sequences=self.num_sample,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature_t,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )

        return {
            "partial_output": partial_output,
            "stopper": stopper,
            "prompt_len": prompt_len,
        }

    @torch.inference_mode()
    def _pack_last_token_hidden_states(self, partial_hidden_states, stage2_device):
        """
        partial_hidden_states:
            length T
            each item: tuple(layer tensors)
            each tensor shape: [N, cur_seq_len, H]

        return:
            hs_tensor: [T, L, N, H]
        """
        output_len = len(partial_hidden_states)
        layer_num = len(partial_hidden_states[0])

        hs_tensor = []
        for pos in range(output_len):
            hs_pos = partial_hidden_states[pos]
            per_pos = []
            for j in range(layer_num):
                per_pos.append(hs_pos[j][:, -1, :].detach().to(stage2_device, dtype=torch.float32))
            hs_tensor.append(torch.stack(per_pos, dim=0))

        hs_tensor = torch.stack(hs_tensor, dim=0)
        return hs_tensor

    @torch.inference_mode()
    def stage2_select_winner(self, partial_output, stopper):
        partial_sequences = partial_output.sequences
        partial_hidden_states = partial_output.hidden_states
        output_len = len(partial_hidden_states)

        if partial_sequences.size(0) == 0:
            raise RuntimeError("ST-BoN stage-1 generation returned no sequences.")

        if output_len == 0:
            return {
                "winner_idx": 0,
                "tau": 0,
                "vote_indices": [0],
                "coe_features_all_steps": [],
                "pairwise_scores_all_steps": [],
            }

        tau = stopper.tau if stopper.tau is not None else output_len
        tau = min(tau, output_len)

        vote_indices = []
        coe_features_all_steps = []
        pairwise_scores_all_steps = []

        stage2_device = self.runtime_device if (self.stage2_on_gpu and torch.cuda.is_available()) else torch.device("cpu")

        hs_tensor = self._pack_last_token_hidden_states(partial_hidden_states, stage2_device)  # [T, L, N, H]
        hs_cumsum = hs_tensor.cumsum(dim=0)

        start_pos = max(0, output_len - tau)

        for end_pos in range(start_pos + 1, output_len + 1):
            hs_mean_layers = hs_cumsum[end_pos - 1] / float(end_pos)  # [L, N, H]

            coe_features_this_step = self.coe_calculator.compute_features_batch(hs_mean_layers)
            sample_scores = self.coe_calculator.compute_pairwise_scores(coe_features_this_step)

            best_idx = int(torch.argmin(sample_scores).item())
            vote_indices.append(best_idx)

            coe_features_all_steps.append(coe_features_this_step.detach().cpu().tolist())
            pairwise_scores_all_steps.append(sample_scores.detach().cpu().tolist())

        vote_counter = Counter(vote_indices)
        max_vote = max(vote_counter.values())
        winner_candidates = [k for k, v in vote_counter.items() if v == max_vote]
        winner_idx = min(winner_candidates)

        del hs_tensor
        del hs_cumsum

        return {
            "winner_idx": winner_idx,
            "tau": tau,
            "vote_indices": vote_indices,
            "coe_features_all_steps": coe_features_all_steps,
            "pairwise_scores_all_steps": pairwise_scores_all_steps,
        }

    @torch.inference_mode()
    def stage3_continue_winner(self, partial_output, winner_idx, prompt_len):
        partial_sequences = partial_output.sequences

        winner_seq = partial_sequences[winner_idx:winner_idx + 1].to(self.runtime_device)
        partial_len = partial_sequences.size(1) - prompt_len
        remain_tokens = max(self.max_output_token - partial_len, 0)

        if remain_tokens > 0:
            final_output = self.model.generate(
                input_ids=winner_seq,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.terminators,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                max_new_tokens=remain_tokens,
                output_scores=True,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature_t,
                do_sample=True,
            )
            final_sequences = final_output.sequences
        else:
            final_sequences = partial_sequences[winner_idx:winner_idx + 1]

        return {
            "final_sequences": final_sequences,
            "partial_len": partial_len,
        }

    @torch.inference_mode()
    def decode(self, input_ids):
        stage1_start = time.time()
        stage1_result = self.stage1_sample_until_stop(input_ids)
        stage1_end = time.time()
        stage1_time = round(stage1_end - stage1_start, 4)

        partial_output = stage1_result["partial_output"]
        stopper = stage1_result["stopper"]
        prompt_len = stage1_result["prompt_len"]

        stage2_start = time.time()
        stage2_result = self.stage2_select_winner(partial_output, stopper)
        stage2_end = time.time()
        stage2_time = round(stage2_end - stage2_start, 4)

        winner_idx = stage2_result["winner_idx"]

        stage3_start = time.time()
        stage3_result = self.stage3_continue_winner(partial_output, winner_idx, prompt_len)
        stage3_end = time.time()
        stage3_time = round(stage3_end - stage3_start, 4)

        final_sequences = stage3_result["final_sequences"]
        inference_time = round(stage1_time + stage2_time + stage3_time, 4)

        output_text, final_output_len = self.decode_single_sequence(
            final_sequences[0],
            prompt_len,
        )

        meta = {
            "output_length": final_output_len,
            "inference_time": inference_time,
            "stage1_time": stage1_time,
            "stage2_time": stage2_time,
            "stage3_time": stage3_time,
            "c": stopper.c,
            "tau": stage2_result["tau"],
            "stop_step": stopper.stop_step,
        }

        return {
            "output_text": output_text,
            "meta": meta,
        }