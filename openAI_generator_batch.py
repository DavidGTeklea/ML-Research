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
    "agent_instrument": r"/ix1/xli/dgt12/AgentLocationInstrument5.csv",
    "agent_patient": r"/ix1/xli/dgt12/AgentLocationPatient61.csv",
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
For the verb "{verb}", list exactly five distinct, prototypical scenarios encountered in everyday life, each with unique {roles_list}. 
Use this format:

<number>. Agent: <agent>; Patient: <patient>; Instrument: <instrument>; Location: <location>
Sentence: "<a sentence that includes all above roles>"

Some roles that may be used include agent (who/what performs the action; must be specific and unique across examples), 
patient (who/what is the recipient of the action; must differ in each case), 
instrument (the means of performing the action), and 
location (where/direction of the action).
Each scenario sentence must include the exact verb "{verb}" as a standalone word in the sentence. 
Each scenario must only have 1 verb, and it must be the verb {verb} (e.g. "Alex grabbed his surfboard and headed out to ride the waves in the ocean." is unacceptable).
Another BAD SCENARIO: The home cook grates nutmeg to enhance the flavor.   # contains "grates" + "enhance"
Do not use a synonym for the verb (e.g., “pirouette” for “dance”). 
Do not nomalize verbs (e.g. "She didn't get much sleep last night is bad" for "She tried poorly to sleep last night").
Do not use phrasal/compound variants (e.g., if {verb} = “sleep”, do not produce “fall asleep,” “oversleep,” “sleep in”).
Avoid descriptive adjectives. 
Avoid repeating agents and avoid generic terms (e.g., "thing," "place")—opt for vivid details.
Finally, state which scenario number (1–5) is best and explain why. A number rating from a scale of 1–10 
should be given to each scenario, and there should be an average among the 5 scenarios for each verb. 
The rating should be done in reference to plausibility of scenario, as well as whether the roles (agent, location, patient, instrument) are 
actually being used in the scenario generation.
"""

# ==== Determine role set ====
if args.file == "agent_instrument":
    roles = ["Agent", "Instrument", "Location"]
elif args.file == "agent_patient":
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
