import random

# Config
INPUT_FILE = "extracted_sentences_with_verbs.txt"
OUTPUT_FILE = "sampled_scenarios_for_annotation.txt"
SAMPLE_SIZE = 100
SEED = 42  # reproducibility

# Load verb-sentence pairs
pairs = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or "|" not in line:
            continue
        verb, sentence = [part.strip() for part in line.split("|", 1)]
        pairs.append((verb, sentence))

# Safety check
if SAMPLE_SIZE > len(pairs):
    raise ValueError(
        f"Requested sample size ({SAMPLE_SIZE}) exceeds available pairs ({len(pairs)}). "
        f"Available: {len(pairs)}"
    )

# Sample randomly
random.seed(SEED)
sampled = random.sample(pairs, SAMPLE_SIZE)

# Write to plain text file (verb and sentence together)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for verb, sentence in sampled:
        f.write(f"{verb} | {sentence}\n")

print(f"[INFO] Wrote {SAMPLE_SIZE} verb-sentence pairs to '{OUTPUT_FILE}'")
