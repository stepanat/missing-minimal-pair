"""
Get log-probabilities of all attribute/group combinations per stereotype (track also num tokens
for length normalization).

Input: data_{lang}_FINAL_CLEAN.json
Output: data_{lang}_FINAL_SCORED.json
 
Usage:
    python scoring/get_log_probs.py --data_dir ./output/ 
"""
import argparse
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import json
import gc
import torch
import os

MODEL_BASENAME_TO_PATH = {
    "aya_32b": "CohereLabs/aya-expanse-32b",
    "llama_8b_base": "meta-llama/Llama-3.1-8B",
    "llama_8b_instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "llama_70b": "RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8",
    "mistral_base": "mistralai/Mistral-Nemo-Base-2407",
    "mistral_instruct": "mistralai/Mistral-Nemo-Instruct-2407",
    "glm_base": "zai-org/glm-4-9b-hf",
    "glm_instruct": "zai-org/glm-4-9b-chat-hf",
    "gpt_oss_20b": "openai/gpt-oss-20b",
}

################   Load model and set sampling params
def load_model(model_path, tp=1):
    model = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.95,
        max_model_len=512,
        max_num_seqs=32,
        enforce_eager=True,
    )
    tokenizer = model.get_tokenizer()
    return model, tokenizer

def get_sampling_params(temp=1.0, max_tok=1, prompt_lp=1):
    sampling_params = SamplingParams(
        temperature = temp,  # should play around with this, maybe results will vary depending on temp
        max_tokens = max_tok,
        prompt_logprobs = prompt_lp,
    )
    return sampling_params

#### Get LPs (LP(att|group)) for all attribute and group pairs
def get_log_likelihood_for_groups_and_atts_batch(model, sampling_params, tokenizer, gg_list, a1, a2, lang):
    # generate output
    out_dict = {gg: {a1: {}, a2: {}} for gg in gg_list}
    sents = []
    groups_and_atts = []
    for gg in gg_list:
        for att in a1, a2:
            out_dict[gg][att] = {}
            if lang == "zh":
                sentence = f"{gg}{att}"
            else:
                sentence = f"{gg} {att}"
            sents.append(sentence)
            groups_and_atts.append((gg, att))
    # process all outputs in batch
    outputs = model.generate(sents, sampling_params)
    # process each output - store in out_dict
    for ii, output in enumerate(outputs):
        gg, att = groups_and_atts[ii]
        prompt_logprobs = output.prompt_logprobs
        prefix_tokens = tokenizer.encode(gg)
        prompt_tokens = output.prompt_token_ids
        # for debugging/inspecting:
        # print(sents[ii])
        # print(f"prompt_tokens: {prompt_tokens}")
        # print(f"prefix_tokens: {prefix_tokens}")
        log_likelihood = 0.0
        num_toks = 0
        for att_tok_id in range(len(prefix_tokens), len(prompt_logprobs)):
            tok = prompt_tokens[att_tok_id]
            logprobs = prompt_logprobs[att_tok_id]
            if logprobs and tok in logprobs:
                # print(f"tok: '{tokenizer.decode(tok)}' and likelihood: {logprobs[tok].logprob:.2f}")
                log_likelihood += logprobs[tok].logprob
                num_toks += 1
        out_dict[gg][att]["LL"] = log_likelihood
        out_dict[gg][att]["num_tokens"] = num_toks
        # print(f"LL({att}|{gg}): {avg_log_likelihood:.2f}")
    # print(out_dict)
    return out_dict

def main():
    parser = argparse.ArgumentParser(description="Compute log-likelihoods of a1/a2 continuations for each group.")
    parser.add_argument("--data_dir", default="./output/",
                        help="Directory with data_{lang}_FINAL_CLEAN.json files from automatic_qa.py")
    parser.add_argument("--models", nargs="+", default=list(MODEL_BASENAME_TO_PATH.keys()),
                        choices=list(MODEL_BASENAME_TO_PATH.keys()))
    parser.add_argument("--languages", nargs="+", default=["en", "es", "ru", "zh"])
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism size")
    args = parser.parse_args()
    ##### RUNNING LOG PROB CALCULATIONS ON THE FINAL DATA FILES
    for model_name in args.models:
        print(f"\n=== Loading Model: {model_name} ===")
        model, tokenizer = load_model(
            MODEL_BASENAME_TO_PATH[model_name],
            tp=args.tp,
        )
        sampling_params = get_sampling_params()
        for lang in args.languages:
            in_path = os.path.join(args.data_dir, f"data_{lang}_FINAL_CLEAN.json")
            out_path = os.path.join(args.data_dir, f"data_{lang}_FINAL_SCORED.json")
            src_path = out_path if os.path.exists(out_path) else in_path
            if not os.path.exists(src_path):
                print(f"[{lang}] {in_path} not found. Skipping.")
                continue
            with open(src_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                groups = item["groups"]
                all_v_keys = sorted([k for k in item.keys() if k.startswith("v")], key=lambda k: int(k[1:]))
                for v_key in all_v_keys:
                    a1 = item[v_key]["a1"]
                    a2 = item[v_key]["a2"]
                    ll_data = get_log_likelihood_for_groups_and_atts_batch(
                        model, sampling_params, tokenizer, groups, a1, a2, lang
                    )
                    for group in groups:
                        if group not in item[v_key]:
                            item[v_key][group] = {}
                        if model_name not in item[v_key][group]:
                            item[v_key][group][model_name] = {}
                        item[v_key][group][model_name]["a1_lp"] = ll_data[group][a1]["LL"]
                        item[v_key][group][model_name]["a1_len"] = ll_data[group][a1]["num_tokens"]
                        item[v_key][group][model_name]["a2_lp"] = ll_data[group][a2]["LL"]
                        item[v_key][group][model_name]["a2_len"] = ll_data[group][a2]["num_tokens"]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Successfully saved to {out_path}")
        del model
        destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()