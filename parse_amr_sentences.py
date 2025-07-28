import sys
import json
from transition_amr_parser.parse import AMRParser

# Initialize the AMR parser
amr_parser = AMRParser.from_pretrained('AMR3-structbart-L')

if len(sys.argv) != 2:
    print("Usage: python parse_amr_sentences.py <sentences_file>")
    sys.exit(1)

sent_file = sys.argv[1]
with open(sent_file, 'r', encoding='utf-8') as f:
    for line in f:
        sentence = line.strip()
        if not sentence:
            continue
        print(f"\nSentence: {sentence}")
        try:
            tokens, _ = amr_parser.tokenize(sentence)
            annots, machines = amr_parser.parse_sentence(tokens)
            amr_graph = machines.get_amr().to_penman(jamr=False, isi=True)
            print(amr_graph)
            # detect roles
            roles = {
                "Agent":      ":ARG0" in amr_graph,
                "Patient":    ":ARG1" in amr_graph,
                "Instrument": ":instrument" in amr_graph,
                "Location":   ":location" in amr_graph,
            }
            print("Roles present:", roles)
        except Exception as e:
            print(f"Error parsing '{sentence}': {e}")