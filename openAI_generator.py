# -*- coding: utf-8 -*-
import os
import sys
import argparse
import pandas as pd
import openai

# ==== Configuration ====
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    sys.exit("Error: OPENAI_API_KEY not set in environment.")

# Predefined list of supported CSV files
INPUT_FILES = {
    "agent_location": r"./AgentLocation51.csv",
    "agent_location_instrument": r"./AgentLocationInstrument5.csv",
    "agent_location_patient": r"./AgentLocationPatient61.csv",
    "all_roles": r"./all_roles60.csv",
}

# ==== CLI Arguments ====
parser = argparse.ArgumentParser(description="Generate role-based scenarios using OpenAI API in batches.")
parser.add_argument(
    "--file",
    choices=INPUT_FILES.keys(),
    help="Which CSV to process. Omit to process ALL datasets."
)
parser.add_argument(
    "--start", type=int, default=0,
    help="Start index of verbs to process (0-based)."
)
parser.add_argument(
    "--end", type=int, default=None,
    help="End index (exclusive) of verbs to process."
)
parser.add_argument(
    "--chunk-id", type=int, default=None,
    help="(Unused when running all) Optional chunk ID for logging."
)
args = parser.parse_args()

# ==== Prompt Builder ====
def build_prompt(verb, roles):
    roles_list = ", ".join(roles)
    return f"""
    
You are generating role-based scenarios. 

For the verb "{verb}", create **exactly 5 distinct, everyday scenarios**, each with unique {roles_list}.

Roles:
- **Agent** (who/what performs the action; must be specific and unique across examples; DO NOT use proper names)
- **Patient** (who/what is the recipient of the action; must differ in each case)
- **Instrument** (the means of performing the action)
- **Location** (always include a **setting location**, typically introduced by 'in', 'at', or 'on'; not just a target location)

**Rules:**
- Each sentence must contain **only the verb "{verb}" in progressive form** (one simple verb, do not use synonyms of "{verb}", do not use two verbs in the same clause).
- **No phrasal/compound verbs** (e.g., "fish for", "cut out," "cut off").
- Avoid nominalization (e.g. "She didn't get much sleep last night" is bad; say instead "She is sleeping poorly").
- Write each scenario so it could be turned directly into an illustration or photo. Use concrete, imageable details (colors, textures, objects) and avoid vague language.
- Sentence must include **{roles_list}**.

Follow this structure exactly.

==== Verb: cut ====
1. Agent: chef; Patient: carrots; Instrument: sharp knife; Location: restaurant kitchen
Sentence: "The chef is cutting the carrots with a sharp knife at the restaurant kitchen."

2. Agent: barber; Patient: customer's hair; Instrument: stainless steel scissors; Location: barbershop
Sentence: "At the barbershop, the barber is cutting the customer's hair with stainless steel scissors."

3. Agent: tailor; Patient: blue fabric; Instrument: fabric shears; Location: sewing studio
Sentence: "The tailor is cutting the blue fabric with fabric shears in the sewing studio."

4. Agent: gardener; Patient: rose bushes; Instrument: pruning shears; Location: backyard garden
Sentence: "The gardener is cutting the rose bushes with pruning shears on the backyard patio."

5. Agent: surgeon; Patient: abdominal tissue; Instrument: surgical scalpel; Location: operating room
Sentence: "In the operating room, the surgeon is cutting the abdominal tissue with a surgical scalpel."

---

"""

def roles_for(dataset_key: str):
    if dataset_key == "agent_location_instrument":
        return ["Agent", "Instrument", "Location"]
    elif dataset_key == "agent_location_patient":
        return ["Agent", "Patient", "Location"]
    elif dataset_key == "all_roles":
        return ["Agent", "Patient", "Instrument", "Location"]
    else:
        return ["Agent", "Location"]

def output_path_for(dataset_key: str) -> str:
    if dataset_key == "all_roles":
        return "/ix1/xli/dgt12/outputs/verb_outputs_all_roles.txt"
    elif dataset_key == "agent_location_patient":
        return "/ix1/xli/dgt12/outputs/verb_outputs_agent_location_patient.txt"
    elif dataset_key == "agent_location_instrument":
        return "/ix1/xli/dgt12/outputs/verb_outputs_agent_location_instrument.txt"
    elif dataset_key == "agent_location":
        return "/ix1/xli/dgt12/outputs/verb_outputs_agent_location.txt"
    # Fallback in case you add new keys later
    return f"/ix1/xli/dgt12/outputs/verb_outputs_{dataset_key}.txt"

def process_dataset(dataset_key: str):
    input_path = INPUT_FILES[dataset_key]
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(input_path)
    else:
        sys.exit(f"Unsupported file: {input_path}")

    verbs = df.iloc[1:, 0].dropna().astype(str).tolist()  # skip header row 0
    verbs = verbs[args.start:args.end]

    if not verbs:
        print(f"[{dataset_key}] No verbs to process in range {args.start}:{args.end}")
        return

    out_path = output_path_for(dataset_key)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"\n=== Processing '{dataset_key}' verbs {args.start}:{args.end} "
          f"({len(verbs)} total) -> {out_path} ===\n")

    with open(out_path, "w", encoding="utf-8") as fout:
        for i, verb in enumerate(verbs, start=1):
            print(f"[{dataset_key}] [{i}/{len(verbs)}] Generating for: {verb}")
            try:
                prompt = build_prompt(verb, roles_for(dataset_key))
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.7,
                    n=1,
                )
                out = resp.choices[0].message.content.strip()
                fout.write(f"==== Verb: {verb} ====\n")
                fout.write(out + "\n\n")
                fout.flush()
            except Exception as e:
                print(f"[{dataset_key}] Error on '{verb}': {e}")
                fout.write(f"==== Verb: {verb} ====\n")
                fout.write(f"ERROR: {e}\n\n")
                fout.flush()

    print(f"[{dataset_key}] Done writing: {out_path}")

# ==== Run one or all ====
datasets_to_run = [args.file] if args.file else list(INPUT_FILES.keys())
for key in datasets_to_run:
    process_dataset(key)
