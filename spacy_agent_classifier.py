import json
import spacy
from nltk.corpus import wordnet as wn

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Try to use WordNet to check animacy
def is_wordnet_animate(word):
    for syn in wn.synsets(word, pos=wn.NOUN):
        if any("person" in lemma.name() or "organism" in lemma.name() for lemma in syn.hypernyms()):
            return True
    return False

def is_likely_animate_v2(token):
    # Entity type-based check
    if token.ent_type_ in {"PERSON", "ORG", "NORP"}:
        return True
    # POS + dependency + WordNet check
    if token.pos_ == "NOUN" and token.dep_ in {"nsubj", "nsubjpass"}:
        return is_wordnet_animate(token.lemma_)
    return False

# File paths
input_file = "potential_agents.jsonl"
output_file = "confirmed_agents.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        entry = json.loads(line)
        sentence = entry["sentence"]
        concepts = entry["concepts"]

        doc = nlp(sentence)
        found_animate = False

        for token in doc:
            if is_likely_animate_v2(token):
                found_animate = True
                break

        result = {"sentence": sentence, "confirmed_agent": found_animate}
        f_out.write(json.dumps(result) + "\n")
        print(f"✓ Checked: {sentence[:50]}... → confirmed_agent = {found_animate}")
