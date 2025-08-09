import sys
import json
from transition_amr_parser.parse import AMRParser
import penman
import re
from pathlib import Path

# Load confirmed agents from classifier
CONFIRMED_AGENT_FILE = "confirmed_agents.jsonl"
confirmed_agents_map = {}
if Path(CONFIRMED_AGENT_FILE).exists():
    with open(CONFIRMED_AGENT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                confirmed_agents_map[obj["sentence"]] = obj.get("confirmed_agent", False)
            except Exception:
                continue

# Save noun candidates if no ARG0 detected
EXTRACT_CONCEPTS = True
concept_dump_file = "potential_agents.jsonl"

def extract_concept_candidates(g):
    var_to_concept = {}
    edges = {}
    for src, rel, tgt in g.triples:
        if rel == ":instance":
            var_to_concept[src] = tgt
        else:
            edges.setdefault(src, []).append((rel, tgt))

    candidates = []
    root_var = g.triples[0][0]
    visited = set()

    def recurse(var, depth=0):
        if var in visited or depth > 4:
            return
        visited.add(var)
        concept = var_to_concept.get(var)
        if concept:
            candidates.append(concept)
        for rel, tgt in edges.get(var, []):
            if rel in {":ARG1"} and tgt in var_to_concept:
                recurse(tgt, depth + 1)

    recurse(root_var)
    return candidates

def fallback_roles_recursive(amr_graph_str, sentence):
    g = penman.decode(amr_graph_str)
    roles_present = {
        "Agent": False,
        "Patient": False,
        "Instrument": False,
        "Location": False
    }

    for _, role, _ in g.triples:
        if role == ":ARG0":
            roles_present["Agent"] = True
        elif role == ":ARG1":
            roles_present["Patient"] = True
        elif role == ":instrument":
            roles_present["Instrument"] = True
        elif role == ":location":
            roles_present["Location"] = True

    # No Agent → extract candidates
    if not roles_present["Agent"] and EXTRACT_CONCEPTS:
        concepts = extract_concept_candidates(g)
        with open(concept_dump_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sentence": sentence, "concepts": concepts}) + "\n")

    # Override if confirmed from classifier
    if not roles_present["Agent"] and confirmed_agents_map.get(sentence, False):
        print("External classifier confirmed agent.")
        roles_present["Agent"] = True

    return roles_present

# Initialize the AMR parser
amr_parser = AMRParser.from_pretrained('AMR3-structbart-L')

if len(sys.argv) != 2:
    print("Usage: python parse_amr_sentences.py <sentences_file>")
    sys.exit(1)

sent_file = sys.argv[1]
output_file = sent_file.rsplit('.', 1)[0] + '_amr_output_v2.txt'

with open(sent_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        sentence = line.strip()
        if not sentence:
            continue

        f_out.write(f"\nSentence: {sentence}\n")
        print(f"\nSentence: {sentence}")
        try:
            tokens, _ = amr_parser.tokenize(sentence)
            annots, machines = amr_parser.parse_sentence(tokens)
            amr_graph = machines.get_amr().to_penman(jamr=False, isi=True)

            f_out.write(amr_graph + "\n")
            print(amr_graph)

            roles = fallback_roles_recursive(amr_graph, sentence)

            role_str = f"Roles present: {roles}\n"
            f_out.write(role_str)
            print(role_str, end='')

        except Exception as e:
            err_msg = f"Error parsing '{sentence}': {e}\n"
            f_out.write(err_msg)
            print(err_msg)
