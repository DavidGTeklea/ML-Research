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
    "agent_instrument": r"./AgentLocationInstrument5.csv",
    "agent_patient": r"./AgentLocationPatient61.csv",
    "all_roles": r"./all_roles60.csv",
}

# ==== CLI Argument ====
parser = argparse.ArgumentParser(description="Generate role-based scenarios using OpenAI API.")
parser.add_argument(
    "--file",
    choices=INPUT_FILES.keys(),
    help="Which CSV to process. If omitted, all files will be processed in sequence."
)
args = parser.parse_args()

# ==== Prompt Builder ====
def build_prompt(verb, roles):
    roles_list = ", ".join(roles)
    return f"""
For the verb "{verb}", list exactly five distinct scenarios, each with unique {roles_list}. Use this format:

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

# ==== Run OpenAI on Verbs ====
keys_to_process = [args.file] if args.file else list(INPUT_FILES.keys())
for key in keys_to_process:
    input_path = INPUT_FILES[key]
    output_path = f"/ix1/xli/dgt12/verb_outputs_{key}.txt"
    print(f"\n=== Processing '{key}' from {input_path} -> {output_path} ===\n")

    # Decide role set
    if key == "agent_instrument":
        roles = ["Agent", "Instrument", "Location"]
    elif key == "agent_patient":
        roles = ["Agent", "Patient", "Location"]
    elif key == "all_roles":
        roles = ["Agent", "Patient", "Instrument", "Location"]
    else:
        roles = ["Agent", "Location"]

    # Load verbs
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(input_path)
    else:
        print(f"Skipping unsupported file: {input_path}")
        continue
    verbs = df.iloc[1:, 0].dropna().astype(str).tolist()

    # Generate scenarios and write cleanly to .txt
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

                # Write to file with a clear section per verb
                fout.write(f"==== Verb: {verb} ====\n")
                fout.write(out + "\n\n")

            except Exception as e:
                print(f"Error on '{verb}': {e}")

    print(f"Done writing: {output_path}")
