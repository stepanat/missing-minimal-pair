from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.distributed.parallel_state import destroy_model_parallel
import json
import gc
import re
import argparse
import os

# create a map of where you expect to load (local or HF for each generator model)
MODEL_BASENAME_TO_PATH = {
    "qwen3_32b": "Qwen/Qwen3-32B",
    "gemma4_31b": "google/gemma-4-31B-it",
}

SYSTEM_PROMPT = (
    "You are an expert multilingual NLP assistant helping with academic research on stereotypes. "
    "Generate a semantic opposite (COUNTER-STEREOTYPE) of a stereotype.\n\n"

    "CONSTRAINTS:\n"
    "1. Keep the TARGET GROUP unchanged.\n"
    "2. Preserve the sentence structure.\n"
    "3. Maintain the polarity of the stereotype:\n"
    "   - If the input stereotype is AFFIRMATIVE (contains no negation), the counter-stereotype must also be affirmative (e.g., 'are lazy' -> 'are hardworking').\n"
    "   - If the input stereotype is NEGATIVE (contains 'not', 'no', 'не', 'لا', '不'), the counter-stereotype must also be a negation, achieving the opposite meaning by negating a negative concept (e.g., 'do NOT understand' -> 'do NOT struggle with').\n"
    "4. Use antonyms or semantically opposite attributes.\n"
    "5. Keep output length close to input length.\n"
    "6. Respond ONLY with a valid JSON object containing a single 'stereo_alt' key."
)

def user_task_paraphrase(stereo, group):
    return f"""
Here are examples across different languages demonstrating strict Polarity Mirroring.

[Example 1 - English: AFFIRMATIVE (+) -> AFFIRMATIVE (+)]
STEREOTYPE: Millennials are lazy.
TARGET GROUP: Millennials
JSON OUTPUT:
{{
  "stereo_alt": "Millennials are hardworking."
}}

[Example 2 - Spanish: NEGATIVE (-) -> NEGATIVE (-)]
STEREOTYPE: Los ancianos no entienden la tecnología.
TARGET GROUP: Los ancianos
JSON OUTPUT:
{{
  "stereo_alt": "Los ancianos no tienen ninguna dificultad con la tecnología."
}}

[Example 3 - Russian: NEGATIVE (-) -> NEGATIVE (-)]
STEREOTYPE: Айтишники не умеют общаться с людьми.
TARGET GROUP: Айтишники
JSON OUTPUT:
{{
  "stereo_alt": "Айтишники не испытывают трудностей в общении с людьми."
}}

[Example 4 - Chinese: AFFIRMATIVE (+) -> AFFIRMATIVE (+)]
STEREOTYPE: 年轻人只关心玩手机。
TARGET GROUP: 年轻人
JSON OUTPUT:
{{
  "stereo_alt": "年轻人只关心努力工作。"
}}

[Example 5 - Arabic: NEGATIVE (-) -> NEGATIVE (-)]
STEREOTYPE: الشباب لا يتحملون المسؤولية.
TARGET GROUP: الشباب
JSON OUTPUT:
{{
  "stereo_alt": "الشباب لا يتهربون من المسؤولية."
}}

Now, generate the single counter-stereotype for the input below.
Requirements:
- Must begin with "{group}".
- Must match the input language.
- Must obey the affirmative/negative polarity match.
- Try not to change the total number of words. 
- Return ONLY a JSON object with the key "stereo_alt".

STEREOTYPE: {stereo}
TARGET GROUP: {group}
JSON OUTPUT:
"""

def format_prompt(stereo, group):
    return SYSTEM_PROMPT + user_task_paraphrase(stereo, group)

################   Load model and set sampling params
def load_model(model_path, tp = 1):
    model = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        max_num_seqs=128,
        enforce_eager=False,
    )
    tokenizer = model.get_tokenizer()
    return model, tokenizer

# recommended gemma 4 params:
def get_sampling_params(temp=1.0, max_tok=1024, top_p=0.95, top_k=64):
    json_schema = {
        "type": "object",
        "properties": {
            "stereo_alt": {"type": "string"}
        },
        "required": ["stereo_alt"],
        "additionalProperties": False
    }
    sampling_params = SamplingParams(
        temperature = temp,  # should play around with this, maybe results will vary depending on temp
        max_tokens = max_tok,
        top_p=top_p,
        top_k=top_k,
        structured_outputs=StructuredOutputsParams(json=json_schema)
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

def strip_group_name(text: str, group: str) -> str:
    """Removes the group name from the start of the string to isolate the attribute."""
    if text.startswith(group):
        return text[len(group):].strip()
    return text.strip()

# function for prompting LLM to return counter-stereotype
def generate_counter_stereos_batch(model, tokenizer, sampling_params, stereo_group_pairs):
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
            results.append({"stereo_alt": "[FAILED_TO_PARSE]"})
    return results

# function for preparing flat list of steretoype variations for batching
def flatten_stereos(stereo_data):
    """Dynamically flattens all available vX keys into a batchable list."""
    flat_list = []
    for item in stereo_data:
        group = item['groups'][0]
        # Find 'v0', 'v1', etc. dynamically
        v_keys = [k for k in item.keys() if k.startswith('v')]
        for v_key in v_keys:
            att = item[v_key]["a1"]
            stereo = f"{group} {att}"
            flat_list.append((stereo, group))
    return flat_list

def load_stereos_from_path(stereo_path):
    with open(stereo_path, encoding="utf-8") as f:
        stereo_data = json.load(f)
    return stereo_data

# helper function to get alternate attribute from a dict item
def get_a2_from_stereo(llm_output, group):
    """Safely extracts the attribute, defaulting to empty if parsing failed."""
    raw_text = llm_output.get("stereo_alt", "")
    return strip_group_name(raw_text, group)

def run_model(model_basename,
              languages,
              input_dir="./output/",
              tp=2):
    model_path = MODEL_BASENAME_TO_PATH[model_basename]
    # Load model and sampling params once
    model, tokenizer = load_model(model_path, tp=tp)
    sampling_params = get_sampling_params(max_tok=4096, temp=0.5)
    # Process each language dynamically
    for lang in languages:
        print(f"\n--- PROCESSING: {lang.upper()} ---")
        filepath = os.path.join(input_dir, f"data_{lang}_{model_basename}.json")
        out_filepath = filepath  # will dynamically fill in a2 in-place
        try:
            stereo_data = load_stereos_from_path(filepath)
        except FileNotFoundError:
            print(f"File not found for {lang}. Skipping.")
            continue
        flat_stereos = flatten_stereos(stereo_data)
        # Call LLM
        all_responses = generate_counter_stereos_batch(model, tokenizer, sampling_params, flat_stereos)
        # Map responses back to the JSON structure
        counter = 0
        for item in stereo_data:
            group = item["groups"][0]
            v_keys = [k for k in item.keys() if k.startswith('v')]
            for v_key in v_keys:
                att = get_a2_from_stereo(all_responses[counter], group)
                item[v_key]["a2"] = att
                counter += 1 
        # Save updated data safely (ensure_ascii=False for multilinguality)
        with open(out_filepath, "w", encoding="utf-8") as f:
            json.dump(stereo_data, f, ensure_ascii=False, indent=4)
            print(f"Saved generated alternates in {out_filepath}")
    # Memory cleanup
    del model
    destroy_model_parallel()
    gc.collect()

# ==========================================
# EXECUTION
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Generate alternate (counter-stereotype) attributes.")
    parser.add_argument("--data_dir", default="./output/", help="Directory from where to read output of generate_paraphrases.py")
    parser.add_argument("--models", nargs="+", default=["gemma4_31b", "qwen3_32b"],
                        choices=list(MODEL_BASENAME_TO_PATH.keys()))
    parser.add_argument("--languages", nargs="+", default=["en", "ru", "ar", "es", "fr", "zh"])
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism size")
    args = parser.parse_args()
    for model_basename in args.models:
        run_model(model_basename, args.languages, input_dir=args.data_dir, tp=args.tp)

if __name__ == "__main__":
    main()
