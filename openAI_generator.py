# -*- coding: utf-8 -*-
import os
import sys
import argparse
import pandas as pd
import json
import openai

# ==== Configuration ====  
# Ensure your API key is set in the environment:
#    export OPENAI_API_KEY="sk-..."
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

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate scenarios for verbs using OpenAI API based on input CSV.")
parser.add_argument(
    "--file",
    choices=INPUT_FILES.keys(),
    help="Which CSV to process. If omitted, all files will be processed in sequence."
)
args = parser.parse_args()

def build_prompt(verb, roles):
    # Compose a numbered template for exactly five scenarios
    roles_list = ", ".join(roles)
    template = [
        f"{i}. " + "; ".join([f"{r}: <...>" for r in roles]) + "\n   Sentence: \"<a sentence that includes all above roles>\""
        for i in range(1, 6)
    ]
    block = "\n\n".join(template)
    return f"""
For the verb "{verb}", list exactly five distinct scenarios, each with unique {roles_list}. Use this exact format:

{block}

Some roles that may be used include agent (who/what performs the action; must be specific and unique across examples), 
patient (who/what is the recipient of the action; must differ in each case), 
instrument (the means of performing the action), and 
location (where/direction of the action).
Each scenario sentence must include the exact verb \"{verb}\" as a standalone word in the sentence. Do not use a synonym (e.g., “pirouette” for “dance”). 
Avoid descriptive adjectives. 
Avoid repeating agents and avoid generic terms (e.g., "thing," "place")—opt for vivid details.
Finally, state which scenario number (1–5) is best and explain why. A number rating from a scale of 1-10 
should be given to each scenario, and there should be an average among the 5 scenarios for each verb. 
The rating should be done in reference to plausaibility of scenario, as well as whether the roles (agent, location, patient, instrument) are 
actually being used in the scenario generation. """

# Main processing loop
keys_to_process = [args.file] if args.file else list(INPUT_FILES.keys())
for key in keys_to_process:
    INPUT_PATH = INPUT_FILES[key]
    base = key
    OUTPUT_PATH = f"/ix1/xli/dgt12/verb_outputs_{base}.jsonl" # TODO <- update this to be outside of llama_job folder
    print(f"\n=== Processing '{base}' from {INPUT_PATH} -> {OUTPUT_PATH} ===\n")

    # Determine roles based on filename key
    if key == "agent_instrument":
        roles = ["Agent", "Instrument", "Location"]
    elif key == "agent_patient":
        roles = ["Agent", "Patient", "Location"]
    elif key == "all_roles":
        roles = ["Agent", "Patient", "Instrument", "Location"]
    else:
        roles = ["Agent", "Location"]
    
    # Load verbs
    ext = os.path.splitext(INPUT_PATH)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(INPUT_PATH)
    elif ext in (".xls", ".xlsx"):
        df = pd.read_excel(INPUT_PATH)
    else:
        print(f"Skipping unsupported file: {INPUT_PATH}")
        continue
    verbs = df.iloc[1:, 0].dropna().astype(str).tolist()

    # Generate and save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
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
                print("Full output:\n", out, "\n")

                # Write a readable JSONL entry: preserve newlines in the file
                entry = {"verb": verb, "response": out}
                line = json.dumps(entry, ensure_ascii=False)
                # Replace escaped newlines with real ones for readability
                line = line.replace('\\n', '\n')
                fout.write(line + "\n\n")
            except Exception as e:
                print(f"Error on '{verb}': {e}")
    print(f"Finished processing '{base}'. Output at {OUTPUT_PATH}\n")
