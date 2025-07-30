#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import openai
import re

# ==== Configuration & Env Check ====  
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    sys.exit("Error: OPENAI_API_KEY not set in environment.")

# Predefined input CSVs
INPUT_FILES = {
    "agent_location": "/ix1/xli/dgt12/AgentLocation51.csv",
    "agent_instrument": "/ix1/xli/dgt12/AgentLocationInstrument5.csv",
    "agent_patient": "/ix1/xli/dgt12/AgentLocationPatient61.csv",
    "all_roles": "/ix1/xli/dgt12/all_roles60.csv",
}

# Prompt builder
def build_prompt(verb, roles):
    roles_list = ", ".join(roles)
    blocks = []
    for i in range(1, 6):
        parts = [f"{r}: <describe {r.lower()}>" for r in roles]
        blocks.append(
            f"{i}. " + "; ".join(parts) +
            "\n   Sentence: \"<a sentence including all above roles>\""
        )
    body = "\n\n".join(blocks)
    return f"""
For the verb \"{verb}\", list exactly five distinct very common scenarios, each with unique roles such as {roles_list}. 
Each sentence must use the verb "\{verb}\" in its inflected verbal form, not as a noun. 
The verb must function as the main action of the sentence (e.g., “The samurai bowed...” is acceptable, but “The butler gave a bow...”  is not). 
Do not use any construction that turns the verb into a noun.
Use this format:
{body}
Some roles that may be used include agent (who/what performs the action; must be specific and unique across examples), 
patient (who/what is the recipient of the action; must differ in each case), 
instrument (the means of performing the action), and 
location (where/direction of the action).
Each scenario sentence must include the exact verb \"{verb}\" as a standalone word in the sentence. Do not use a synonym (e.g., “pirouette” for “dance”). 
Avoid descriptive adjectives. 
Avoid repeating agents and avoid generic terms (e.g., "thing," "place")—opt for vivid details.
Finally, state which scenario (1–5) is best, explain why, provide a 1–10 rating for each, and compute the average rating.
""".strip()

# Main driver
def main():
    parser = argparse.ArgumentParser(
        description="Generate scenario sentences via OpenAI GPT without AMR parsing"
    )
    parser.add_argument(
        "--file", choices=INPUT_FILES.keys(),
        help="CSV key to process; if omitted, all keys are processed"
    )
    args = parser.parse_args()

    keys = [args.file] if args.file else list(INPUT_FILES.keys())
    out_path = "generated_sentences.txt"
    with open(out_path, 'w', encoding='utf-8') as fout:
        for key in keys:
            input_path = INPUT_FILES[key]
            # load verbs
            ext = os.path.splitext(input_path)[1].lower()
            df = pd.read_csv(input_path) if ext == ".csv" else pd.read_excel(input_path)
            verbs = df.iloc[1:, 0].dropna().astype(str).tolist()

            roles = {
                "agent_instrument": ["Agent","Instrument","Location"],
                "agent_patient":    ["Agent","Patient","Location"],
                "all_roles":        ["Agent","Patient","Instrument","Location"],
            }.get(key, ["Agent","Location"])

            for verb in verbs:
                prompt = build_prompt(verb, roles)
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=800, temperature=0.7,
                )
                raw = resp.choices[0].message.content.strip()
                # extract sentences
                blocks = re.findall(r"(\d+\.[\s\S]*?)(?=\n\d+\.|\Z)", raw)[:5]
                for block in blocks:
                    m = re.search(r'Sentence:\s*"([^"]+)"', block)
                    sentence = m.group(1) if m else None
                    if sentence:
                        fout.write(sentence + "\n")
    print(f"Wrote sentences to {out_path}")

if __name__ == '__main__':
    main()
