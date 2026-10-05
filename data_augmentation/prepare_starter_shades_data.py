import pandas as pd
import stanza
import spacy_stanza
import torch
import re
import json
import os
import argparse

BIAS_SHADES_SPLITS = {
    'ar': 'by_language/ar.csv',
    'bn': 'by_language/bn.csv',
    'de': 'by_language/de.csv',
    'en': 'by_language/en.csv',
    'es': 'by_language/es.csv',
    'fr': 'by_language/fr.csv',
    'hi': 'by_language/hi.csv',
    'it': 'by_language/it.csv',
    'mr': 'by_language/mr.csv',
    'nl': 'by_language/nl.csv',
    'pl': 'by_language/pl.csv',
    'pt_br': 'by_language/pt_br.csv',
    'ro': 'by_language/ro.csv',
    'ru': 'by_language/ru.csv',
    'zh': 'by_language/zh.csv',
    'zh_hant': 'by_language/zh_hant.csv'
}

def get_subject_token(doc):
    """
    Finds the first noun phrase
    """
    demographic_deps = {
        "nsubj", "nsubj:pass", "csubj", "nsubj:outer", "nsubj:cop",
        "iobj", "obl", "obl:arg", "nmod", "dislocated", "vocative", "dep",
        "obj"
    }
    valid_pos = {"NOUN", "PROPN", "PRON", "PART", "NUM"}
    # Scan up to index 5 to tolerate leading verbs (Arabic VSO) and modals
    for token in doc:
        if token.pos_ in valid_pos and token.dep_ in demographic_deps:
            if token.i <= 5: 
                return token
    has_verb = any(t.pos_ in ("VERB", "AUX") for t in doc)
    if not has_verb:
        root = next((t for t in doc if t.dep_.lower() == "root"), None)
        if root and root.pos_ in valid_pos:
            return root
    return None

def extract_group_and_att(doc, original_text):
    """
    Extracts Group and Attribute from parsed doc.
    Relies on spacy_stanza models having processed the input text as a doc
    and uses get_subject_token to find the token corresponding to starting NP.
    """
    if not isinstance(original_text, str) or not doc:
        return str(original_text).strip(), ""
    dash_match = re.search(r'\s+([-–—])\s+', original_text)
    if dash_match:
        dash_idx = dash_match.start()
        return original_text[:dash_idx].strip(), original_text[dash_idx:].strip()
    subj_token = get_subject_token(doc)
    if not subj_token:
        return original_text.strip(), ""
    # find the rightmost edge of the dep parsing tree found for subj token
    right_token = subj_token.right_edge
    if right_token.i >= len(doc) - 2:
        right_token = subj_token
    group_span = doc[0 : right_token.i + 1]
    escaped_tokens = [re.escape(t.text) for t in group_span]
    pattern = r'^\s*' + r'\s*'.join(escaped_tokens)
    match = re.match(pattern, original_text)
    if match:
        exact_group = match.group(0)
        attr_text = original_text[match.end():]
        return exact_group.strip(), " ".join(attr_text.split())
    group_text = group_span.text.strip()
    right_part = "".join([t.text_with_ws for t in doc if t.i > right_token.i])
    return group_text, " ".join(right_part.split())

def filter_consistent_attributes(df, group_col='index', attr_col='Attribute'):
    """
    Filters the dataframe to keep groups where the Attribute is identical,
    AND strictly removes any rows where the parsing failed (empty Attributes).
    This is necessary because if there is leakage about the group identity into
    the attribute (e.g. if verbs are gendered and thereby reveal gender of subject group),
    our method doesn't work (it needs strict 1:1 comparison).
    """
    df[attr_col] = df[attr_col].fillna("").astype(str).str.strip()
    is_consistent = df.groupby(group_col)[attr_col].transform('nunique') == 1
    is_not_empty = df[attr_col] != ""
    return df[is_consistent & is_not_empty].copy()

def main():
    parser = argparse.ArgumentParser(description="Parse BiasShades stereotypes into group/attribute format.")
    parser.add_argument("--output_dir", default="./output/", help="Directory to save CSV and JSON outputs")
    parser.add_argument("--languages", nargs="+", default=["en", "ru", "ar", "zh", "es", "fr"],
                        help="Language codes to process")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ########################################################################
    # LOAD SPACY-STANZA MODELS FOR THE FOLLOWING LANGUAGES TO DO DEP PARSING
    languages = args.languages

    # PYTORCH/STANZA WORKAROUND
    # PyTorch newer versions break Stanza in some versions, so we need a workaround
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    print("Downloading Stanza models (only required once)...")
    for lang in languages:
        stanza.download(lang)
    print("Loading Stanza models safely...")
    processors_config = 'tokenize,pos,lemma,depparse'
    nlps = {
        lang: spacy_stanza.load_pipeline(lang, processors=processors_config) 
        for lang in languages
    }
    torch.load = _original_torch_load

    ########################################################################
    # LOAD STARTING BIAS SHADES DATA
    all_dfs = {}
    for lang in languages:
        all_dfs[lang] = pd.read_csv("hf://datasets/LanguageShades/BiasShades/" + BIAS_SHADES_SPLITS[lang])
        print(all_dfs[lang].shape) # each of shape (728, 12)
    for lang, df_lang in all_dfs.items():
        all_dfs[lang] = df_lang[df_lang["type"] == "declaration"].copy()
    
    ########################################################################
    # FOR EACH STEREOTYPE, WE SEPARATE THE GROUP FROM ATTRIBUTE
    for lang, df in all_dfs.items():
        print(f"Processing {lang} in batches...")
        target_col = f'{lang}_biased_sentences'
        texts = df[target_col].fillna("").astype(str).tolist()
        groups = []
        attributes = []
        # nlp.pipe processes the texts in batches
        for doc, text in zip(nlps[lang].pipe(texts, batch_size=50), texts):
            g, a = extract_group_and_att(doc, text)
            print(f"GROUP:{g}\tATT:{a}")
            groups.append(g)
            attributes.append(a)
        df['Group'] = groups
        df['Attribute'] = attributes
        all_dfs[lang] = df

    ########################################################################
    # FOR EACH LANG, keep only the stereotypes with have identical attribute wordings across the different groups
    for lang in all_dfs:
        original_size = len(all_dfs[lang])
        all_dfs[lang] = filter_consistent_attributes(all_dfs[lang])
        new_size = len(all_dfs[lang])
        print(f"{lang}: Dropped {original_size - new_size} mismatched rows.")

    # Print the final resulting unique stereotypes per language
    for lang, df in all_dfs.items():
        idx_col = 'index' if 'index' in df.columns else 'Index'
        # Count the unique stereotype IDs
        unique_count = df[idx_col].nunique()
        print(f"{lang.upper():<5} | {unique_count} stereotypes remaining")

    ########################################################################
    # SAVE FINAL DATA
    # save the data as CSV
    for lang, df in all_dfs.items():
        out_path = os.path.join(output_dir, f"data_{lang}.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")

    # process into JSONs
    all_dfs = {}
    for lang in languages:
        all_dfs[lang] = pd.read_csv(os.path.join(output_dir, f"data_{lang}.csv"))

    for lang, df in all_dfs.items():
        lang_data = []
        idx_col = 'index' if 'index' in df.columns else 'Index'
        target_col = f'{lang}_biased_sentences' if f'{lang}_biased_sentences' in df.columns else 'Biased Sentences'
        for idx, group_df in df.groupby(idx_col):
            bias_type = group_df['bias_type'].iloc[0]
            base_attribute = group_df['Attribute'].iloc[0]
            groups = group_df['Group'].tolist()
            reference_sentence = group_df[target_col].iloc[0]
            # Build the exact dictionary structure
            lang_data.append({
                "index": int(idx),
                "bias_type": bias_type,
                "groups": groups,
                "stereotype": reference_sentence,
                "v0": {
                    "a1": base_attribute,
                    "a2": ""
                }
            })
        # Save as JSON per language
        out_path = os.path.join(output_dir, f"data_{lang}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(lang_data, f, ensure_ascii=False, indent=4)
            
    print("JSON conversion complete!")
    

if __name__ == "__main__":
    main()
