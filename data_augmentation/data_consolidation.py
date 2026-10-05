"""
Consolidate generated paraphrases and alternates across different LLMs used for generation.
For each language:
1) merge output of all models (all data_{lang}_{model}.json files)
2) drop obvious failures (parse failures, empty, identical a1/a2, subject leaks into attribute)
3) optionally drop grammar errors flagged by LanguageTool
4) deduplicate variants and write data_{lang}_merged.json.

Note: v0 (the original stereotyped attribute) is taken from the first model file that is read,
in the order given by --models.
 
Usage:
    python data_augmentation/data_consolidation.py --data_dir ./output/

    # If running LanguageTool, as in paper, need --grammar_check flag, but note that for this
    # need to have Java (e.g. apt-get install -y openjdk-17-jre-headless)

    python data_augmentation/data_consolidation.py --data_dir ./output/ --grammar_check

Helper qa_sample_attributes() function can be used to randomly sample some generated attributes
to figure out the best rule-based approach for filtering.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
import language_tool_python
import argparse
import re
import os
from tqdm import tqdm

####################################################################################################
# Helpers and env vars for using language_tool_python to do grammar checks (requires Java)
os.environ["JAVA_TOOL_OPTIONS"] = "-Xmx2g -Xms512m"

LANG_MAP = {
    "en": "en-US",
    "es": "es",
    "ru": "ru",
    "zh": "zh-CN",
}

# Cache language tools globally so Java servers don't spin up/down in loops
TOOLS_CACHE = {}
TOOLS_USAGE_COUNT = {}
MAX_CHECKS_BEFORE_RESTART = 500

def get_tool(lang_code):
    if lang_code in TOOLS_CACHE and TOOLS_USAGE_COUNT[lang_code] >= MAX_CHECKS_BEFORE_RESTART:
        tqdm.write(f"\n[SYSTEM] Recycling JVM for [{lang_code}] after {MAX_CHECKS_BEFORE_RESTART} checks to clear RAM")
        TOOLS_CACHE.pop(lang_code).close()  # recycle the JVM to free memory
    if lang_code not in TOOLS_CACHE:
        tqdm.write(f"\n[SYSTEM] Starting fresh LanguageTool JVM for [{lang_code}].")
        for tool in TOOLS_CACHE.values():
            tool.close()
        TOOLS_CACHE.clear()
        tool = language_tool_python.LanguageTool(lang_code)
        tool.enabled_categories = {'GRAMMAR', 'SYNTAX'}
        tool.enabled_rules_only = True  # execute ONLY the whitelisted categories
        tool.disabled_categories = set()
        tool.disabled_rules = set()
        TOOLS_CACHE[lang_code] = tool
        TOOLS_USAGE_COUNT[lang_code] = 0
        tqdm.write(f"[SYSTEM] JVM for [{lang_code}] successfully started.")
    # update usage counter
    TOOLS_USAGE_COUNT[lang_code] += 1
    return TOOLS_CACHE[lang_code]
####################################################################################################

# helper function to print some sample data per language to figure out which bugs (e.g. subject repetition)
# to automatically process
def qa_sample_attributes(data_dir: str, sample_size: int = 10):
    input_path = Path(data_dir)
    lang_files = defaultdict(list)
    # Group files by language prefix
    for file in input_path.glob("data_*_*.json"):
        parts = file.stem.split("_")
        if len(parts) >= 2:
            lang = parts[1]
            lang_files[lang].append(file)
    for lang, files in sorted(lang_files.items()):
        print(f"\n==================================================")
        print(f"LANGUAGE: {lang.upper()} | Files scanned: {[f.name for f in files]}")
        print(f"==================================================")
        all_variations = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                groups = entry.get("groups", [])
                stereotype = entry.get("stereotype", "")
                for key, val in entry.items():
                    if key.startswith("v") and isinstance(val, dict):
                        a1 = val.get("a1", "")
                        a2 = val.get("a2", "")
                        if a1 or a2:
                            all_variations.append((stereotype, groups, key, a1, a2, file.name))
        # Randomly sample from collected variations
        k = min(sample_size, len(all_variations))
        samples = random.sample(all_variations, k)
        for i, (st, grp, v_key, a1, a2, fname) in enumerate(samples, 1):
            print(f"\n--- Sample {i}/{k} [Source: {fname} -> {v_key}] ---")
            print(f"Groups:     {grp}")
            print(f"Stereotype: {st}")
            print(f"a1 (Prem):  {a1}")
            print(f"a2 (Hyp):   {a2}")

####################################################################################################
# HELPERS FOR SANITY CHECKING THAT GENERATED ALTERNATES AND PARAPHRASES ARE PARSEABLE AND USABLE
CAPITAL_PATTERNS = {
    "en": r"^[A-Z]",
    "es": r"^[A-ZÁÉÍÓÚÜÑ]",   # Latin uppercase plus Spanish diacritics and Ñ
    "ru": r"^[А-Я]",
}
SUBSTRING_LEAK_LANGS = {"es", "ru"}

def clean_key(text: str) -> str:
    # Keeps letters/numbers across scripts, strips punctuation, spaces, and underscores
    return re.sub(r'[^\w\u4e00-\u9fff\u0400-\u04FF\u0600-\u06FF]|_', '', text).lower()

def check_well_formed_attribute(a1: str, a2: str, groups: list, lang: str) -> tuple[bool, str]:
    """
    Language-specific checks: we found that a capitalized attribute usually means the model
    repeated the subject, sometimes in a different grammatical form.
    For es/ru, use a plain substring check also catches
    inflected group names; for en this was too aggressive, so it's skipped.
    """
    a1, a2 = a1.strip(), a2.strip()
    # VERY BASIC SANITY CHECKS for empty attributes, a2==a1, parsing errors
    if not a1 or not a2: return False, "EMPTY"
    a1_clean, a2_clean = clean_key(a1), clean_key(a2)
    a1_lower, a2_lower = str(a1).lower(), str(a2).lower()
    if ("failed" in a1_lower and "parse" in a1_lower) or ("failed" in a2_lower and "parse" in a2_lower): return False, "FAILED_TO_PARSE"
    if a1_clean == a2_clean: return False, "IDENTICAL_A1_A2"
    if len(a1_clean) == 0 or len(a2_clean) == 0:
        return False, "A1_OR_A2_EMPTY"
    # General check (all languages): group name appears as a whole word in the attribute
    for g in groups:
        g_clean = re.sub(r'^[^\w]+|[^\w]+$', '', g.strip().lower())
        if not g_clean:
            continue
        if lang == "zh":
            g_c = clean_key(g)
            if len(g_c) >= 2 and (g_c in a1_clean or g_c in a2_clean):
                return False, f"SUBJECT_LEAK_[{g.strip()}]"
        elif len(g_clean) >= 2:
            pattern = r'\b' + re.escape(g_clean) + r'\b'
            if re.search(pattern, a1_lower) or re.search(pattern, a2_lower):
                return False, f"SUBJECT_LEAK_[{g.strip()}]"
    # Lang-specific extra checks
    if lang not in CAPITAL_PATTERNS:
        return True, "VALID"
    if len(a1) < 3 or len(a2) < 3:
        return False, "EMPTY_A1_OR_A2"
    # catch cases where group name appears anywhere in attribute, this was too
    # aggressive for EN ("men" could appear in a lot of different words), whereas
    # for ES and RU this extra check is needed
    if lang in SUBSTRING_LEAK_LANGS:
        for g in groups:
            g_clean = g.strip().lower()
            if g_clean and (g_clean in a1_lower or g_clean in a2_lower):
                return False, f"SUBJECT_LEAK_[{g_clean}]"
    if re.match(CAPITAL_PATTERNS[lang], a1):
        return False, f"{lang.upper()}_A1_CAPITALIZED_SO_SUBJ_LEAKAGE"
    if re.match(CAPITAL_PATTERNS[lang], a2):
        return False, f"{lang.upper()}_A2_CAPITALIZED_SO_SUBJ_LEAKAGE"
    return True, "VALID"

def check_validity_with_lang_tool(a1: str, a2: str, groups: list, lang: str) -> tuple[bool, str]:
    tool = get_tool(LANG_MAP[lang])
    # check that the sentences are valid with the first group only
    g = groups[0]
    opt_space = " " if lang != "zh" else ""
    sent1 = f"{g}{opt_space}{a1}"
    sent2 = f"{g}{opt_space}{a2}"
    for match in (tool.check(sent1) + tool.check(sent2)):
        issue_type = getattr(match, 'rule_issue_type', getattr(match, 'ruleIssueType', ''))
        rule_id = getattr(match, 'rule_id', getattr(match, 'ruleId', ''))
        if issue_type in ["style", "whitespace", "locale-violation"]:
            continue
        if rule_id in ["UPPERCASE_SENTENCE_START", "COMMA_PARENTHESIS_WHITESPACE"]:
            continue
        return False, f"GRAMMAR_[{rule_id}]"
    return True, "VALID"

################################################################################
# DATA CONSOLIDATION MAIN FUNCTION:
# MERGE LLM GENERATOR OUTPUT, DEDUP, REMOVE EMPTY/FAILED TO PARSE, OPTIONALLY CHECK GRAMMAR
def consolidate(data_dir: str, lang: str, models: list, grammar_check=False):
    """
    Merges the generations of all generator models for one language, drops malformed
    variants (subject leaks, parse failures, identical a1/a2, grammar errors), and
    deduplicates across models. v0 is taken from the first model file found.
    Writes data_{lang}_merged.json.
    """
    in_path = Path(data_dir)
    merged, bug_counts, examples_by_reason = {}, defaultdict(int), {}
    total_raw = total_dedup = 0
    for model_name in models:
        filepath = in_path / f"data_{lang}_{model_name}.json"
        if not filepath.exists():
            print(f"[{lang}] File not found, skipping: {filepath}")
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in tqdm(data, desc=f"{lang.upper()} [{model_name}]", leave=False):
            idx = entry["index"]
            if idx not in merged:
                merged[idx] = {k: entry.get(k) for k in ["index", "bias_type", "groups", "stereotype", "v0"]}
                merged[idx]["pool"] = {}
            groups, pool = merged[idx]["groups"], merged[idx]["pool"]
            var_keys = sorted([k for k in entry if re.fullmatch(r"v\d+", k) and k != "v0"], key=lambda x: int(x[1:]))
            for k in var_keys:
                v = entry[k]
                total_raw += 1
                a1 = v.get("a1", "").replace('\r', ' ').replace('\n', ' ').strip()
                a2 = v.get("a2", "").replace('\r', ' ').replace('\n', ' ').strip()
                # do cheap checks first (remove cases where a1==a2, invalid/empty attribute, subject leak, etc) 
                valid, reason = check_well_formed_attribute(a1, a2, groups, lang)
                # optionally run a language tool (default is off because requires Java and many issues can
                # be caught with basic checks)
                if valid and grammar_check:
                    valid, reason = check_validity_with_lang_tool(a1, a2, groups, lang)
                if not valid:
                    bug_counts[reason] += 1
                    examples_by_reason.setdefault(reason, (a1, a2))
                    continue
                key = (clean_key(a1), clean_key(a2))
                if key in pool:
                    total_dedup += 1
                    continue
                pool[key] = {**v, "a1": a1, "a2": a2}
    final_data = []
    for item in merged.values():
        if not item["pool"]:
            continue
        clean_entry = {k: item[k] for k in ["index", "bias_type", "groups", "stereotype", "v0"]}
        for i, var_data in enumerate(item["pool"].values(), start=1):
            clean_entry[f"v{i}"] = var_data
        final_data.append(clean_entry)

    out_file = in_path / f"data_{lang}_merged.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    total_post = sum(len(item["pool"]) for item in merged.values())
    avg = total_post / len(final_data) if final_data else 0
    print(f"{'LANG':<5} | {'PRE-STEREO':<10} | {'POST-STEREO':<11} | {'AVG VARIANTS':<12} | {'VAR PRE':<10} | {'BUGS':<6} | {'DEDUPED':<7} | {'VAR POST'}")
    print(f"{lang.upper():<5} | {len(merged):<10} | {len(final_data):<11} | {avg:<12.2f} | {total_raw:<10} | {sum(bug_counts.values()):<6} | {total_dedup:<7} | {total_post}")
    if bug_counts:
        print(f"  --- Bugs Caught ({lang}) ---")
        for reason, count in sorted(bug_counts.items(), key=lambda x: x[1], reverse=True):
            a1, a2 = examples_by_reason[reason]
            print(f"  * {reason}: {count} items dropped")
            print(f"    Example -> a1: {a1} | a2: {a2}")
    print("-" * 90)


def main():
    parser = argparse.ArgumentParser(description="Merge generator outputs, drop malformed variants, deduplicate.")
    parser.add_argument("--data_dir", default="./output/",
                        help="Directory with data_{lang}_{model}.json files")
    parser.add_argument("--models", nargs="+", default=["gemma4_31b", "qwen3_32b"],
                        help="Generator models to merge (v0 is taken from the first one found)")
    parser.add_argument("--languages", nargs="+", default=["en", "ru", "zh", "es"],
                        choices=list(LANG_MAP.keys()))
    parser.add_argument("--grammar_check", action="store_true",
                    help="Optionally drop variants with LanguageTool grammar errors (requires Java)")
    args = parser.parse_args()
    for lang in args.languages:
        consolidate(args.data_dir, lang, args.models, args.grammar_check)
 
 
if __name__ == "__main__":
    main()
