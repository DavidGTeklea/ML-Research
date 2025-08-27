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
    "agent_location": r"/ix1/xli/dgt12/AgentLocation51.csv",
    "agent_location_instrument": r"/ix1/xli/dgt12/AgentLocationInstrument5.csv",
    "agent_location_patient": r"/ix1/xli/dgt12/AgentLocationPatient61.csv",
    "all_roles": r"/ix1/xli/dgt12/all_roles60.csv",
}

# ==== CLI Arguments ====
parser = argparse.ArgumentParser(description="Generate role-based scenarios using OpenAI API in batches.")
parser.add_argument(
    "--file",
    choices=INPUT_FILES.keys(),
    required=True,
    help="Which CSV to process."
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
    help="Optional chunk ID (for logging and output naming in array jobs)."
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

# ==== Determine role set ====
if args.file == "agent_location_instrument":
    roles = ["Agent", "Instrument", "Location"]
elif args.file == "agent_location_patient":
    roles = ["Agent", "Patient", "Location"]
elif args.file == "all_roles":
    roles = ["Agent", "Patient", "Instrument", "Location"]
else:
    roles = ["Agent", "Location"]

# ==== Load verbs ====
input_path = INPUT_FILES[args.file]
ext = os.path.splitext(input_path)[1].lower()
if ext == ".csv":
    df = pd.read_csv(input_path)
elif ext in (".xls", ".xlsx"):
    df = pd.read_excel(input_path)
else:
    sys.exit(f"Unsupported file: {input_path}")

verbs = df.iloc[1:, 0].dropna().astype(str).tolist()

# Apply slice for batch processing
verbs = verbs[args.start:args.end]

if not verbs:
    sys.exit(f"No verbs to process in range {args.start}:{args.end}")

# ==== Output naming ====
chunk_suffix = f"_chunk{args.chunk_id}" if args.chunk_id is not None else f"_{args.start}-{args.end}"
output_path = f"/ix1/xli/dgt12/outputs/verb_outputs_{args.file}{chunk_suffix}.txt"

# ==== Generate scenarios ====
print(f"\n=== Processing '{args.file}' verbs {args.start}:{args.end} "
      f"({len(verbs)} total) -> {output_path} ===\n")

with open(output_path, "w", encoding="utf-8") as fout:
    for i, verb in enumerate(verbs, start=1):
        print(f"[{i}/{len(verbs)}] Generating for: {verb}")
        try:
            prompt = build_prompt(verb, roles)
            resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.7,
                n=1,
            )
            out = resp.choices[0].message.content.strip()

            # Write with a clear section header
            fout.write(f"==== Verb: {verb} ====\n")
            fout.write(out + "\n\n")
            fout.flush()

        except Exception as e:
            print(f"Error on '{verb}': {e}")
            fout.write(f"==== Verb: {verb} ====\n")
            fout.write(f"ERROR: {e}\n\n")
            fout.flush()

print(f"Done writing: {output_path}")
