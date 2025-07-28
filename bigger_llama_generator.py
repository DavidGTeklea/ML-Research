# -*- coding: utf-8 -*-
import os
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==== Config ====  
MODEL_ID    = "meta-llama/Llama-2-70b-chat-hf"
EXCEL_PATH  = r"/ix1/xli/dgt12//llama_job/NLP_project_verb_list_MWD.xlsx"
OUTPUT_PATH = r"/ix1/xli/dgt12/llama_job/verb_outputs_70b.jsonl"

# ==== Load model and tokenizer via HF cache ====
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True,
    use_auth_token=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",     # auto-shards across all GPUs/CPU
    load_in_8bit=True,     # quantize to 8-bit to save VRAM
    torch_dtype="auto",    # lets bitsandbytes pick fp16 for you
    use_auth_token=True
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ==== Load verbs ====
df = pd.read_excel(EXCEL_PATH)
verbs = df.iloc[1:, 0].dropna().astype(str).tolist()

# ==== Prompt generator ====
def build_prompt(verb):
    return f"""
For the verb "{verb}" (do not include synonyms of this verb), list five distinct, prototypical scenarios encountered in everyday life, each with a unique combination of: agent (who/what performs the action; must be specific and unique across examples), patient (who/what is the recipient of the action; must differ in each case), instrument (the means of performing the action), and location (where/direction of the action). 

Avoid descriptive adjectives. For each scenario, provide specific, concrete examples for all four roles (e.g., not just "a person" but "a toddler"; not just "a room" but "a grocery store") and include a sample sentence where all roles are explicitly named (no implied roles). Avoid repeating agents and avoid generic terms (e.g., "thing," "place")—opt for vivid details. 

Finally, evaluate which of the five examples fits the prompt best and explain why.
"""

# ==== Generate per verb ====
def generate_for_verb(verb):
    prompt = build_prompt(verb)
    result = generator(prompt, max_new_tokens=800, do_sample=True, temperature=0.7)[0]["generated_text"]
    response_only = result[len(prompt):].strip()
    return response_only

# ==== Generate and save ====
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for i, verb in enumerate(verbs):
        print(f"[{i+1}/{len(verbs)}] Generating for: {verb}")
        try:
            response = generate_for_verb(verb)
            print("Sample output:", response, "\n")  
            f.write(json.dumps({"verb": verb, "response": response}) + "\n")
        except Exception as e:
            print(f"Error on '{verb}': {e}")
