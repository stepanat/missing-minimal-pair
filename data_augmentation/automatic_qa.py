"""
Automatic QA of the consolidated data (output of data_consolidation.py).
 
For each language:
1) score paraphrases (v1..vn) against the original stereotype (v0):
   - NLI contradiction in both directions ("contr_score_stereo_premise_para_hyp",
     "contr_score_para_premise_stereo_hyp")
   - LaBSE cosine similarity ("labse_score")
2) score alternates (a2) against attributes (a1) for all of v0..vn:
   - NLI contradiction in both directions ("contr_score_a1_premise_a2_hyp",
     "contr_score_a1_hyp_a2_premise")
3) filter by thresholds:
   - for paraphrase pair (vn a1 and vn a2) to be kept, we require
     max(contr_score_stereo_premise_para_hyp, contr_score_para_premise_stereo_hyp) < 0.8 AND labse_score >= 0.70
   - for alternate pair (vn a1 and vn a2) to be kept, we require
     min(contr_score_a1_premise_a2_hyp, contr_score_a1_hyp_a2_premise) >= 0.8.
   - surviving variants are re-indexed from v0
   - note that the thresholds of 0.8 and 0.7 above are defaults, configure via --contr_threshold and --labse_threshold
4) remove paraphrases identical to v0 and re-index v1..vn
   -> save as data_{lang}_FINAL_CLEAN.json (only the surviving a1/a2 pairs)

Usage:
    python data_augmentation/automatic_qa.py --data_dir ./output/
"""
import argparse
import json
import os
import re
import sys

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# needed to print non-ascii characters in terminal to inspect output
sys.stdout.reconfigure(encoding="utf-8")


#########################################################################
# NLI and LaBSE model loading
#### LOAD NLI MODEL FOR CONTRADICTIONS
def load_nli_model(model_name="joeddav/xlm-roberta-large-xnli"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nli_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    print(f"NLI model loaded on {device}")
    contr_idx = nli_model.config.label2id.get(
        "contradiction", nli_model.config.label2id.get("CONTRADICTION", 0)
    )
    return {"tokenizer": tokenizer, "model": nli_model, "device": device, "contr_idx": contr_idx}

#### LOAD LABSE MODEL FOR SEMANTIC SIMILARITY OF PARAPHRASES
def load_labse_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(
        "sentence-transformers/LaBSE",
        device=device,
        model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
    )

#########################################################################
# HELPERS
def join_group_and_attribute(group, att, lang):
    return f"{group}{att}" if lang == "zh" else f"{group} {att}"
 
 
def get_variant_keys(item, include_v0=True):
    """Returns v0, v1, ... sorted numerically (so v10 comes after v9)."""
    keys = [k for k in item if re.fullmatch(r"v\d+", k) and isinstance(item[k], dict)]
    if not include_v0:
        keys = [k for k in keys if k != "v0"]
    return sorted(keys, key=lambda k: int(k[1:]))
 
 
def get_contradiction_scores_batched(premises, hypotheses, nli):
    """Returns a list of probabilities (0.0 to 1.0) for the contradiction class."""
    inputs = nli["tokenizer"](premises, hypotheses, padding=True, truncation=True,
                              return_tensors="pt").to(nli["device"])
    with torch.no_grad():
        logits = nli["model"](**inputs).logits
    return torch.softmax(logits, dim=-1)[:, nli["contr_idx"]].cpu().tolist()
 
 
def get_bidirectional_contradiction_scores(entries, nli, batch_size):
    """entries: list of (v_dict, premise, hypothesis). Returns list of (premise->hyp, hyp->premise)."""
    scores = []
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        premises = [e[1] for e in batch]
        hypotheses = [e[2] for e in batch]
        fwd = get_contradiction_scores_batched(premises, hypotheses, nli)
        bwd = get_contradiction_scores_batched(hypotheses, premises, nli)
        scores.extend((round(s1, 4), round(s2, 4)) for s1, s2 in zip(fwd, bwd))
    return scores

####################################################################################################
# STEP 1: SCORE PARAPHRASES
def score_labse_in_place(data, lang, labse, batch_size=64):
    entries = []
    for item in data:
        group = item.get("groups", [""])[0]
        stereo_a1 = item.get("v0", {}).get("a1", "")
        if not group or not stereo_a1:
            continue
        base_text = join_group_and_attribute(group, stereo_a1, lang)
        for k in get_variant_keys(item, include_v0=False):
            v_a1 = item[k].get("a1", "")
            if not v_a1:
                continue
            entries.append((item[k], base_text, join_group_and_attribute(group, v_a1, lang)))
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        with torch.no_grad():
            emb_base = labse.encode([e[1] for e in batch], convert_to_tensor=True, normalize_embeddings=True)
            emb_vars = labse.encode([e[2] for e in batch], convert_to_tensor=True, normalize_embeddings=True)
        for (v_dict, _, _), score in zip(batch, (emb_base * emb_vars).sum(dim=-1).cpu().tolist()):
            v_dict["labse_score"] = round(score, 4)

def score_paraphrase_contradictions_in_place(data, lang, nli, batch_size=64):
    entries = []
    for item in data:
        group = item.get("groups", [""])[0]
        stereo_a1 = item.get("v0", {}).get("a1", "")
        for k in get_variant_keys(item, include_v0=False):
            v_dict = item[k]
            entries.append((v_dict,
                            join_group_and_attribute(group, stereo_a1, lang),
                            join_group_and_attribute(group, v_dict["a1"], lang)))
    for (v_dict, _, _), (c1, c2) in zip(entries, get_bidirectional_contradiction_scores(entries, nli, batch_size)):
        v_dict["contr_score_stereo_premise_para_hyp"] = c1
        v_dict["contr_score_para_premise_stereo_hyp"] = c2

####################################################################################################
# STEP 2: SCORE ALTERNATES
def score_contradictions_in_place(data, lang, nli, batch_size=64):
    entries = []
    for item in data:
        group = item.get("groups", [""])[0]
        for k in get_variant_keys(item, include_v0=True):
            v_dict = item[k]
            a1, a2 = v_dict.get("a1", ""), v_dict.get("a2", "")
            entries.append((v_dict,
                            join_group_and_attribute(group, a1, lang),
                            join_group_and_attribute(group, a2, lang)))
    for (v_dict, _, _), (c1, c2) in zip(entries, get_bidirectional_contradiction_scores(entries, nli, batch_size)):
        v_dict["contr_score_a1_premise_a2_hyp"] = c1
        v_dict["contr_score_a1_hyp_a2_premise"] = c2

####################################################################################################
# STEP 3: FILTER BASED ON PARAPHRASE AND ALTERNATE SCORES
def filter_by_labse_and_min_contr_score(data, labse_threshold=0.70, contr_threshold=0.80):
    filtered_data = []
    for entry in data:
        valid_vars = {}
        for k in get_variant_keys(entry, include_v0=True):
            v = entry[k]
            # alternate must contradict the attribute in both directions
            c1, c2 = v.get("contr_score_a1_premise_a2_hyp"), v.get("contr_score_a1_hyp_a2_premise")
            if c1 is None or c2 is None or min(c1, c2) < contr_threshold:
                continue
            if k != "v0":
                # paraphrase must not contradict the original stereotype, and be semantically close
                p1 = v.get("contr_score_stereo_premise_para_hyp", 1.0)
                p2 = v.get("contr_score_para_premise_stereo_hyp", 1.0)
                if max(p1, p2) >= contr_threshold:
                    continue
                if v.get("labse_score", 0) < labse_threshold:
                    continue
            valid_vars[k] = v
        if not valid_vars:
            continue
        clean_entry = {k: entry[k] for k in ["index", "bias_type", "groups", "stereotype"] if k in entry}
        for i, var_data in enumerate(valid_vars.values()):
            clean_entry[f"v{i}"] = var_data
        filtered_data.append(clean_entry)
    return filtered_data

####################################################################################################
# STEP 4: DEDUPLICATE AND REINDEX PARAPHRASES AND ALTERNATES THAT PASSES THE AUTOMATED CUTOFFS
def deduplicate_and_reindex(data):
    """Removes paraphrases whose a1 is identical to v0's a1, then re-indexes v1..vn."""
    for item in data:
        v0_a1 = item.get("v0", {}).get("a1", "")
        var_keys = get_variant_keys(item, include_v0=False)
        valid_vars = [item[k] for k in var_keys if item[k].get("a1", "") != v0_a1]
        for k in var_keys:
            del item[k]
        for idx, var_data in enumerate(valid_vars, start=1):
            item[f"v{idx}"] = var_data
    return data


def clean_models_and_scores(data):
    """Returns a copy of the data with only metadata and a1/a2 per variant (scores removed)."""
    keep_meta = ["index", "bias_type", "groups", "stereotype"]
    cleaned = []
    for item in data:
        new_item = {k: item[k] for k in keep_meta if k in item}
        for k in get_variant_keys(item):
            new_item[k] = {"a1": item[k].get("a1", ""), "a2": item[k].get("a2", "")}
        cleaned.append(new_item)
    return cleaned

####################################################################################################
# MAIN:
def main():
    parser = argparse.ArgumentParser(description="Score consolidated data with NLI + LaBSE and filter by thresholds.")
    parser.add_argument("--data_dir", default="./output/",
                        help="Directory with data_{lang}_merged.json files from data_consolidation.py")
    parser.add_argument("--languages", nargs="+", default=["en", "es", "ru", "zh"])
    parser.add_argument("--contr_threshold", type=float, default=0.80,
                        help="NLI contradiction threshold for both paraphrases and alternates")
    parser.add_argument("--labse_threshold", type=float, default=0.70,
                        help="Minimum LaBSE similarity between a paraphrase and the original stereotype")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
 
    nli, labse = load_nli_model(), load_labse_model()
 
    print(f"{'LANG':<5} | {'STEREOS PRE':<11} | {'STEREOS POST':<12} | {'AVG VARIANTS'}")
    for lang in args.languages:
        in_path = os.path.join(args.data_dir, f"data_{lang}_merged.json")
        clean_path = os.path.join(args.data_dir, f"data_{lang}_FINAL_CLEAN.json")
        if not os.path.exists(in_path):
            print(f"[{lang}] {in_path} not found. Skipping.")
            continue
        with open(in_path, encoding="utf-8") as f:
            data = json.load(f)
 
        score_paraphrase_contradictions_in_place(data, lang, nli, args.batch_size)
        score_labse_in_place(data, lang, labse, args.batch_size)
        score_contradictions_in_place(data, lang, nli, args.batch_size)
 
        final_data = filter_by_labse_and_min_contr_score(data, args.labse_threshold, args.contr_threshold)
        final_data = clean_models_and_scores(deduplicate_and_reindex(final_data))

        with open(clean_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        n_vars = [len(get_variant_keys(e)) for e in final_data]
        avg = sum(n_vars) / len(n_vars) if n_vars else 0
        print(f"{lang.upper():<5} | {len(data):<11} | {len(final_data):<12} | {avg:.2f}")
 
 
if __name__ == "__main__":
    main()

