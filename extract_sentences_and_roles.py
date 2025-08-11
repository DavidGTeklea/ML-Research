# extract_sentences_and_roles.py

import os
import re

INPUT_FILES = [
    "verb_outputs_all_roles.txt",
    "verb_outputs_agent_patient.txt",
    "verb_outputs_agent_instrument.txt",
    "verb_outputs_agent_location.txt",
]

OUTPUT_FILE = "extracted_scenarios_with_roles.txt"

role_keywords = {"agent", "patient", "instrument", "location"}

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for filename in INPUT_FILES:
        if not os.path.exists(filename):
            print(f"[WARN] File not found: {filename}")
            continue

        with open(filename, "r", encoding="utf-8") as f:
            scenario_num = 0
            current_block = []
            for line in f:
                line_stripped = line.strip()

                if re.match(r"^\d+\.\s*Agent:", line_stripped):
                    # New scenario block
                    if current_block:
                        fout.write("\n".join(current_block) + "\n\n")
                        current_block = []

                    scenario_num += 1
                    roles_line = line_stripped
                    current_block.append(f"Scenario {scenario_num}:")
                    # Temporarily hold roles to split cleanly
                    roles = [r.strip() for r in roles_line.split(";") if ":" in r]
                    for role in roles:
                        role_name, role_value = role.split(":", 1)
                        current_block.append(f"{role_name.strip().capitalize()}: {role_value.strip()}")

                elif line_stripped.startswith("Sentence:"):
                    sentence = line_stripped.split("Sentence:", 1)[1].strip().strip('"')
                    if current_block:
                        current_block[0] += f" {sentence}"
            # Add final block if needed
            if current_block:
                fout.write("\n".join(current_block) + "\n\n")

print(f"[DONE] Scenarios with roles written to {OUTPUT_FILE}")
