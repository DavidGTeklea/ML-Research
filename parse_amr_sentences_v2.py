import sys
import json
from transition_amr_parser.parse import AMRParser
import penman
import re 

def is_animate_heuristic(word):
    animate_keywords = {
        "man", "woman", "child", "soldier", "person", "teacher", "doctor", "boy", "girl",
        "agent", "nurse", "student", "officer", "mother", "father", "adult", "human"
    }
    return word.lower() in animate_keywords

def fallback_roles_recursive(amr_graph_str):
    g = penman.decode(amr_graph_str)
    roles_present = {
        "Agent": False,
        "Patient": False,
        "Instrument": False,
        "Location": False
    }

    # Build var → concept map and var → edges
    var_to_concept = {}
    edges = {}

    for src, rel, tgt in g.triples:
        if rel == ":instance":
            var_to_concept[src] = tgt
        else:
            edges.setdefault(src, []).append((rel, tgt))

    # First pass: direct roles
    for src, role, tgt in g.triples:
        if role == ":ARG0":
            roles_present["Agent"] = True
        elif role == ":ARG1":
            roles_present["Patient"] = True
        elif role == ":instrument":
            roles_present["Instrument"] = True
        elif role == ":location":
            roles_present["Location"] = True

    # Fallback agent detection using recursive search
    def recurse_ARG1_chain(var, depth=0, visited=None):
        if visited is None:
            visited = set()
        if var in visited or depth > 4:
            return False
        visited.add(var)

        concept = var_to_concept.get(var, "")
        if is_animate_heuristic(concept):
            print(f"Fallback: treating '{concept}' as Agent (depth={depth})")
            return True

        for rel, tgt in edges.get(var, []):
            if rel == ":ARG1" and tgt in var_to_concept:
                if recurse_ARG1_chain(tgt, depth + 1, visited):
                    return True
        return False

    if not roles_present["Agent"]:
        root_var = g.triples[0][0]  # usually the top concept's variable
        if recurse_ARG1_chain(root_var):
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
            
            roles = fallback_roles_recursive(amr_graph)
            
            role_str = f"Roles present: {roles}\n"
            f_out.write(role_str)
            print(role_str, end='')

        except Exception as e:
            err_msg = f"Error parsing '{sentence}': {e}\n"
            f_out.write(err_msg)
            print(err_msg)

