from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel
import json
import copy
import re
import gc
import os
import argparse

# create a map of where you expect to load (local or HF for each generator model)
MODEL_BASENAME_TO_PATH = {
    "qwen3_32b": "Qwen/Qwen3-32B",
    "gemma4_31b": "google/gemma-4-31B-it",
}

def load_stereos_from_path(stereo_path):
    with open(stereo_path, encoding="utf-8") as f:
        stereo_data = json.load(f)
    return stereo_data

############### DEFINE PROMPTS
SYSTEM_PROMPT_EN = (
    "You are an expert multilingual NLP assistant helping with academic research on bias. "
    "Your task is to generate exactly 10 semantically equivalent paraphrases of a STEREOTYPE.\n\n"
    "CONSTRAINTS:\n"
    "1. Each paraphrase MUST start with the TARGET GROUP and MUST form a grammatically correct, complete sentence.\n"
    "2. Maintain the exact stereotypical meaning. Do NOT sanitize, soften, or correct the bias.\n"
    "3. Preserve the absolute certainty of the statement (no hedging with 'tend to' or 'often').\n"
    "4. Respond ONLY with a raw JSON object containing a 'paraphrases' array of 10 strings."
)

def user_task_paraphrase(stereo, group):
    return f"""
Here are examples across different languages showing the exact expected output.
Note: While these examples show 3 paraphrases for brevity, you MUST generate exactly 10.

[Example - English]
FULL STEREOTYPE: Millennials are lazy.
TARGET GROUP: Millennials
JSON OUTPUT:
{{
  "paraphrases": [
    "Millennials are slackers.",
    "Millennials don't want to work hard.",
    "Millennials have zero work ethic."
  ]
}}

[Example - Spanish]
FULL STEREOTYPE: Los ancianos no entienden la tecnología.
TARGET GROUP: Los ancianos
JSON OUTPUT:
{{
  "paraphrases": [
    "Los ancianos no comprenden la tecnología.",
    "Los ancianos son completamente incapaces de usar aparatos tecnológicos.",
    "Los ancianos se pierden por completo cuando se trata de nuevas tecnologías."
  ]
}}

[Example - Russian]
FULL STEREOTYPE: Айтишники не умеют общаться с людьми.
TARGET GROUP: Айтишники
JSON OUTPUT:
{{
  "paraphrases": [
    "Айтишники не умеют разговаривать с людьми.",
    "Айтишники лишены навыков социального взаимодействия.",
    "Айтишники предпочитают компьютеры живому общению."
  ]
}}

[Example - Chinese]
FULL STEREOTYPE: 年轻人只关心玩手机。
TARGET GROUP: 年轻人
JSON OUTPUT:
{{
  "paraphrases": [
    "年轻人只在乎玩手机。",
    "年轻人整天沉迷于智能手机屏幕。",
    "年轻人把所有时间都花在玩移动设备上。"
  ]
}}

[Example - Arabic]
FULL STEREOTYPE: الشباب لا يتحملون المسؤولية.
TARGET GROUP: الشباب
JSON OUTPUT:
{{
  "paraphrases": [
    "الشباب لا يتحملون الواجبات.",
    "الشباب يفتقرون إلى حس المسؤولية والالتزام.",
    "الشباب يتهربون دائمًا من تحمل أي أعباء أو مسؤوليات."
  ]
}}

STEREOTYPE: {stereo}
TARGET GROUP: {group}
JSON OUTPUT:
"""

def format_prompt(stereo, group):
    return SYSTEM_PROMPT_EN + user_task_paraphrase(stereo, group)

################   Load model and set sampling params
def load_model(model_path, tp = 1):
    model = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.95,
        max_model_len=8192,
        max_num_seqs=16,
        enforce_eager=True,
    )
    return model, model.get_tokenizer()

# recommended sampling params
def get_sampling_params(temp=1.0, max_tok=1024, top_p=0.95, top_k=32, presence_pen=0.0):
    sampling_params = SamplingParams(
        temperature = temp,  # should play around with this, maybe results will vary depending on temp
        max_tokens = max_tok,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_pen,
    )
    return sampling_params

######## Helper function to extract json from markdown
def clean_output_for_json(text: str) -> str:
    text = text.strip()
    # If outputted thinking, remove that block
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    elif "<think>" in text and text.endswith("```"): 
        # Catch edge case where it opened a think block, forgot to close it, but wrote markdown
        text = re.sub(r'<think>.*?(```json|```)', '', text, flags=re.DOTALL)
    # handle markdown
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    # Handle the closing tag
    if text.endswith("```"):
        text = text[:-3]
    # again strip to remove any lingering newlines from inside the code block
    text = text.strip()
    # last resort:
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx+1]
        return text

# if LLM generated group and not just attribute, lets strip it
def clean_generated_predicate(pred: str, target_group: str) -> str:
    pred = pred.strip()
    # Check case-insensitive start (e.g. group is "Boys", model wrote "boys love cars")
    if pred.lower().startswith(target_group.lower()):
        # Slice off the group length, then strip the leading space
        pred = pred[len(target_group):].lstrip()
    return pred

# function for prompting LLM to return 10 paraphrases of the provided stereotype (also giving the group with which the stereotype should start)
def generate_stereo_variants_batch(model, tokenizer, sampling_params, stereo_group_pairs):
    prompts = []
    for stereo, group in stereo_group_pairs:
        content = format_prompt(stereo, group)
        messages = [{"role": "user", "content": content}]
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(prompt_text)
    print(f"Running inference on {len(prompts)} items...")
    outputs = model.generate(prompts, sampling_params)
    results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        cleaned_text = clean_output_for_json(generated_text)
        try:
            parsed_json = json.loads(cleaned_text)
            results.append(parsed_json)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse JSON for output:\n{generated_text}")
            results.append({"paraphrases": ["[FAILED TO PARSE]"] * 10})
    return results

# function for taking input item from data list and returning tuple of stereo, group
def get_stereo_pair_with_group(item):
    group = item['groups'][0]
    return (f"{group} {item['v0']['a1']}", group)

### helper functions to process the results
def extract_variations_list(parsed_obj):
    for key in ["paraphrases", "variations", "paraphrased_variants", "variants", "stereotypes"]:
        if key in parsed_obj:
            return parsed_obj[key]
    return ["[FAILED TO PARSE]"] * 10

# create new data structure and dump the results
def dump_results(output_filename, og_data_list, generated_variations):
    new_data = []
    for item_idx, item in enumerate(og_data_list):
        new_dict = copy.deepcopy(item) 
        stereo_group = item["groups"][0]
        raw_predicates = extract_variations_list(generated_variations[item_idx])
        for par_idx in range(10):
            if par_idx < len(raw_predicates) and raw_predicates[par_idx].strip() and raw_predicates[par_idx] != "[FAILED TO PARSE]":
                safe_predicate = clean_generated_predicate(raw_predicates[par_idx], stereo_group)
            else:
                safe_predicate = "[FAILED TO PARSE]"
            new_dict[f"v{par_idx+1}"] = {"a1": safe_predicate, "a2": ""}
        new_data.append(new_dict)
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    return new_data

############### LOAD LLM AND SAMPLING PARAMS
def run_model(model_basename,
              languages,
              input_dir="./output/", tp=2):
    model_path = MODEL_BASENAME_TO_PATH[model_basename]
    # Load model and sampling params once
    model, tokenizer = load_model(model_path, tp=tp)
    sampling_params = get_sampling_params(max_tok=4096, temp=0.5)
    # Process each language dynamically
    for lang in languages:
        print(f"\n--- PROCESSING: {lang.upper()} ---")
        # Construct paths based on your file naming convention
        filepath = os.path.join(input_dir, f"data_{lang}.json")
        out_filepath = os.path.join(input_dir, f"data_{lang}_{model_basename}.json")
        try:
            stereo_data = load_stereos_from_path(filepath)
        except FileNotFoundError:
            print(f"File not found for {lang} at {filepath}. Skipping.")
            continue
        # Extract pairs for the prompt
        pairs = [get_stereo_pair_with_group(item) for item in stereo_data]
        # Call LLM
        all_responses = generate_stereo_variants_batch(model, tokenizer, sampling_params, pairs)
        # save results
        dump_results(out_filepath, stereo_data, all_responses)
    # Memory cleanup
    del model
    destroy_model_parallel()
    gc.collect()

# --- Execution ---
def main():
    parser = argparse.ArgumentParser(description="Generate paraphrases of stereotypes.")
    parser.add_argument("--data_dir", default="./output/", help="Directory from where to read starter stereo data (parsed into Group, Attribute already)")
    parser.add_argument("--models", nargs="+", default=["gemma4_31b", "qwen3_32b"],
                        choices=list(MODEL_BASENAME_TO_PATH.keys()))
    parser.add_argument("--languages", nargs="+", default=["en", "ru", "ar", "es", "fr", "zh"])
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism size")
    args = parser.parse_args()
    for model_basename in args.models:
        run_model(model_basename, args.languages, input_dir=args.data_dir, tp=args.tp)

if __name__ == "__main__":
    main()

