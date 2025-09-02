# extract_sentences_only.py

import os

INPUT_FILES = [
    "verb_outputs_all_roles.txt",
    "verb_outputs_agent_location_patient.txt",
    "verb_outputs_agent_location_instrument.txt",
    "verb_outputs_agent_location.txt",
]

OUTPUT_FILE = "extracted_sentences_only.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for filename in INPUT_FILES:
        if not os.path.exists(filename):
            print(f"[WARN] File not found: {filename}")
            continue
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("Sentence:"):
                    sentence = line.split("Sentence:", 1)[1].strip().strip('"')
                    fout.write(sentence + "\n")

print(f"[DONE] Sentences saved to {OUTPUT_FILE}")
