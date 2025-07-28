import os
import sys
import argparse
import pandas as pd
import json
import openai
import re
from transition_amr_parser.parse import AMRParser

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

# Initialize the AMR parser (path or name of pretrained model)
amr_parser = AMRParser.from_pretrained('AMR3-structbart-L')  # use correct model name or path

# CLI arguments
parser = argparse.ArgumentParser(
    description="Generate and validate scenarios via OpenAI + AMR parsing."
)
parser.add_argument(
    "--file", choices=INPUT_FILES.keys(),
    help="Key of CSV to process (agent_location, agent_instrument, agent_patient, all_roles)"
)
args = parser.parse_args()

# Prompt builder

def build_prompt(verb, roles):
    roles_list = ", ".join(roles)
    blocks = []
    for i in range(1,6):
        parts = [f"{r}: <describe {r.lower()}>" for r in roles]
        blocks.append(
            f"{i}. " + "; ".join(parts) +
            "\n   Sentence: \"<a sentence including all above roles>\""
        )
    body = "\n\n".join(blocks)
    return f"""
For the verb \"{verb}\", list exactly five distinct scenarios, each with unique {roles_list}. Use this format:

{body}

Finally, state which scenario (1–5) is best, explain why, provide a 1–10 rating for each, and compute the average rating.
""".strip()

# Main driver
def main():
    keys = [args.file] if args.file else list(INPUT_FILES.keys())
    for key in keys:
        input_path = INPUT_FILES[key]
        output_path = f"/ix1/xli/dgt12/verb_outputs_{key}.jsonl"
        print(f"\n=== Processing {key}: {input_path} -> {output_path} ===\n")

        # Select roles
        if key == "agent_instrument": roles = ["Agent","Instrument","Location"]
        elif key == "agent_patient": roles = ["Agent","Patient","Location"]
        elif key == "all_roles":  roles = ["Agent","Patient","Instrument","Location"]
        else: roles = ["Agent","Location"]

        # Load verbs
        ext = os.path.splitext(input_path)[1].lower()
        df = pd.read_csv(input_path) if ext == ".csv" else pd.read_excel(input_path)
        verbs = df.iloc[1:,0].dropna().astype(str).tolist()

        with open(output_path, 'w', encoding='utf-8') as fout:
            for idx, verb in enumerate(verbs, start=1):
                print(f"[{idx}/{len(verbs)}] Generating '{verb}'")
                prompt = build_prompt(verb, roles)
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=800, temperature=0.7, n=1
                )
                text = resp.choices[0].message.content.strip()
                print("GPT →", text.replace("\n"," | "))

                # split on scenario headings 1.,2.,…
                scenarios = re.findall(r"(\d+\.[\s\S]*?)(?=\n\d+\.|\Z)", text)[:5]
                validated = []
                for scen in scenarios:
                    # find the sentence line
                    m = re.search(r'Sentence:\s*"([^"]+)"', scen)
                    sentence = m.group(1) if m else ''

                    if not sentence:
                        print("Skipping empty or malformed sentence block.")
                        continue

                    # parse and print AMR
                    try:
                        tokens, _ = amr_parser.tokenize(sentence)
                        annots, machines = amr_parser.parse_sentence(tokens)
                        amr_graph = machines.get_amr().to_penman(jamr=False, isi=True)
                        print("AMR graph for:", sentence)
                        print(amr_graph)
                    except Exception as e:
                        print(f"AMR parse error for '{sentence}': {e}")
                        amr_graph = ''

                    # check role presence
                    presence = {r: (r.lower() in amr_graph.lower()) for r in roles}
                    print("Roles present:", presence)
                    validated.append({'sentence': sentence, 'roles_present': presence})

                # write JSONL record
                entry = {'verb': verb, 'response': text, 'validation': validated}
                line = json.dumps(entry, ensure_ascii=False).replace('\\n','\n')
                fout.write(line + "\n\n")

        print(f"Finished '{key}'\n")

if __name__=='__main__':
    main()
